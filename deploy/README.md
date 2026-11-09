# Deploying SupplyGraph

One container image runs both the explorer server and the nightly pipeline. A Helm chart
deploys them to Kubernetes. The design is recorded in
[ADR 0011](../docs/adr/0011-kubernetes-single-image-cronjob-publish.md) (deployment),
[0010](../docs/adr/0010-api-write-protection-modes.md) (auth) and
[0012](../docs/adr/0012-observability-freshness-first.md) (observability).

## Container only (no Kubernetes)

```powershell
docker build -t supplygraph:0.8.0 .
docker volume create supplygraph-data
docker run --rm -v supplygraph-data:/data supplygraph:0.8.0 demo-data      # demo snapshot
docker run -d -p 127.0.0.1:8000:8000 -v supplygraph-data:/data --read-only --tmpfs /tmp `
  -e SUPPLYGRAPH_API_TOKEN=change-me supplygraph:0.8.0
# http://127.0.0.1:8000 ; real data instead of the demo:
docker run --rm -v supplygraph-data:/data --env-file .env supplygraph:0.8.0 run-all --skip-graph
```

Without `SUPPLYGRAPH_API_TOKEN`, a container (bound to 0.0.0.0) serves **read-only**: decisions
and Ask are refused. That is deliberate; see ADR 0010.

## Kubernetes (Helm)

```powershell
# local cluster, e.g. kind:
kind create cluster --name supplygraph
kind load docker-image supplygraph:0.8.0 --name supplygraph

helm install sg deploy/helm/supplygraph -n supplygraph --create-namespace `
  --set api.token.existingSecret=supplygraph-token     # a Secret with key api-token
helm test sg -n supplygraph
kubectl -n supplygraph port-forward svc/sg-supplygraph 8000:80
```

What gets deployed:

| Object | Purpose |
|---|---|
| Deployment `…-supplygraph` | `supplygraph serve`; probes `/healthz` (liveness) and `/readyz` (readiness). With `demo.enabled`, an init container writes the demo snapshot if nothing has been published yet. |
| CronJob `…-pipeline` | Nightly `run-all --skip-graph` into `/data/work`. Publishes `/data/warehouse.duckdb` by atomic rename **only if the quality gate passed**. |
| PVC `…-data` | Warehouse, raw downloads, agent transcripts and the analyst audit log. Kept on `helm uninstall`. |
| Secret(s) | API token; optional pipeline secrets (`SEC_USER_AGENT`, `NEO4J_PASSWORD`, `ANTHROPIC_API_KEY`) via `secrets.existingSecret`. |
| Optional | Ingress, ServiceMonitor, PrometheusRule (alerts), `helm test` pod. |

Run the pipeline now instead of waiting for 03:00 UTC:

```powershell
kubectl -n supplygraph create job --from=cronjob/sg-supplygraph-pipeline pipeline-manual
kubectl -n supplygraph logs -f job/pipeline-manual
```

### Values you will want to set

| Value | Default | Notes |
|---|---|---|
| `image.repository` / `image.tag` | `supplygraph` / appVersion | push the image to your registry |
| `api.token.existingSecret` | `""` | without a token the server is read-only |
| `secrets.existingSecret` | `""` | `SEC_USER_AGENT` enables filings + supplier extraction (it contains an email: keep it in a Secret) |
| `pipeline.schedule` | `0 3 * * *` (UTC) | |
| `neo4j.uri` + `pipeline.loadGraph` | off | Neo4j is not part of this chart |
| `metrics.serviceMonitor.enabled` / `metrics.prometheusRule.enabled` | off | need the Prometheus Operator CRDs |
| `persistence.size` / `storageClass` | 5Gi / default | the OpenSanctions list alone is ~80 MB per run |

### Limits (by design, see ADR 0011)

- **One API replica, one node.** The volume is ReadWriteOnce, so updates use `Recreate`, and the
  CronJob has pod affinity to the API pod's node. Scaling out needs RWX or object storage.
- **Reads are not authenticated.** Use ingress auth or keep the service internal if the data
  must not be public.

## Verified

On a kind cluster (Kubernetes v1.37, 2026-11-09):
- `helm lint`, and `helm install` with an API token;
- `helm test` passed: ready, 278 companies, metrics, UI;
- token auth through the Service: 401 without the token, accepted with it;
- one manual pipeline Job against the live sources (GLEIF, OpenSanctions, GDELT; no SEC user
  agent configured) ran `run-all`, passed the quality gate, and published. After that the
  server served the new warehouse without a restart.
