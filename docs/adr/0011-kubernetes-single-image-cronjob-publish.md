# 0011. Kubernetes: one image, CronJob builds and atomically publishes the warehouse

**Status:** Accepted (M8)

## Context
Two things run in production: a long-lived API/UI server, and a nightly pipeline that
downloads sources and rebuilds the warehouse. DuckDB allows one writer, and a reader that
opens the file mid-build would see a partial state or hit a lock (0002). Dagster (M2) is good
for development, backfills and lineage, but running its daemon and webserver in-cluster would
triple the moving parts for one nightly job.

## Decision
- **One image** (`Dockerfile`, multi-stage) for both workloads: the UI built with Node, a
  Python 3.11 slim runtime, and a non-root user (uid 10001). It runs with a read-only root
  filesystem and writable `/data` and `/tmp`, and logs JSON. Dagster is not installed in it.
- **Helm chart** (`deploy/helm/supplygraph`): a Deployment for the server, a CronJob for the
  pipeline, a PVC for `/data`, and optional Ingress, ServiceMonitor and PrometheusRule.
  `helm test` checks readiness, data, metrics and UI.
- **Build, then publish.** The CronJob runs `run-all --skip-graph` in `/data/work`. Only if it
  exits 0, meaning blocking quality checks passed, does it copy the warehouse next to
  `/data/warehouse.duckdb` and `mv` it into place. The rename is atomic, and the server opens
  the file per request, so readers switch to the new data with no downtime and never see a
  failed or partial build.
- **One audit log.** Both workloads set `SUPPLYGRAPH_ACTIONS_LOG=/data/actions/events.jsonl`,
  so the pipeline applies exactly the decisions analysts made in the UI.
- **Demo first.** With `demo.enabled`, an init container writes the committed real-data
  snapshot (`supplygraph demo-data`) when nothing has been published yet. A fresh install is
  usable immediately, and real pipeline output is never overwritten.

## Alternatives considered
- **Dagster in-cluster.** Deferred until there are several pipelines, or a need for backfills
  in production.
- **Shared object storage (S3) for the warehouse.** Removes the RWO constraint, but adds
  credentials and a sync step.

## Consequences
- The data volume is ReadWriteOnce. The API runs one replica with `Recreate` updates, and the
  CronJob has pod affinity to the API pod's node. Scaling out needs RWX storage or object
  storage, which would supersede this record.
- `concurrencyPolicy: Forbid` keeps two runs from racing on `/data/work`.
- Neo4j is not part of the chart. Set `neo4j.uri` and `pipeline.loadGraph` to load an existing
  instance.
- **Verified on kind (v1.37):** install, `helm test`, token auth, and one real pipeline run
  that published (see [deploy/README.md](../../deploy/README.md)).
