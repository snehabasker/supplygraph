# SupplyGraph

**Supplier risk intelligence on an ontology-backed knowledge graph.**

> *"Which of our suppliers are exposed to sanctions, hidden ownership, or disruption — and what should we do about it?"*

SupplyGraph ingests open corporate-ownership, sanctions, SEC filing and news data, maps it into
a typed ontology (objects, links, actions), loads it into Neo4j, scores explainable risk over
the graph, and answers analysts' questions with a Claude agent whose every change is a
human-approved, audited action.

## Architecture

```
GLEIF API ─────┐                                     ┌─► Neo4j knowledge graph ─► GraphRAG agent ─► audited actions
OpenSanctions ─┤                                     │                         └─► Explorer UI (`serve`)
SEC EDGAR ─► supplier extraction ─┼─► raw JSONL/CSV ─► DuckDB ─► DQ checks ─┤
   (10-K/20-F)  (rules | Claude)  │     (bronze)       (silver)              └─► Graph ML risk scores (M5)
GDELT GKG ─────┘
```

| Layer | Tech | Why |
|---|---|---|
| Ingestion | `httpx`, retry + rate limiting | GLEIF allows 60 req/min, SEC 10 req/s; transient failures are retried, 404s are data |
| Extraction | regex baseline, Claude structured output | Supplier relationships from annual reports; every link quotes a verbatim sentence |
| Warehouse | DuckDB | Zero-ops columnar SQL; full-refresh builds are idempotent |
| Quality | SQL checks, error/warn severity | Pipeline refuses to load the graph on blocking failures |
| Graph | Neo4j 5 | Native traversal for ownership chains; uniqueness constraints |

See [docs/ontology.md](docs/ontology.md) for the object/link model.

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env          # then set NEO4J_PASSWORD and SEC_USER_AGENT

docker compose up -d            # Neo4j at http://localhost:7474

supplygraph ingest-gleif        # ~2-3 min (rate-limited)
supplygraph ingest-sanctions    # ~80 MB download
supplygraph ingest-filings      # latest 10-K / 20-F of 12 filers (~60 MB, once)
supplygraph extract-suppliers   # supplier links + GLEIF lookup of each supplier (~2 min)
supplygraph ingest-news         # last 24 h of GDELT (~160 MB streamed, a few MB kept)
supplygraph build
supplygraph resolve
supplygraph score-risk
supplygraph check
supplygraph load-graph
# or: supplygraph run-all
```

On macOS/Linux use `source .venv/bin/activate` and `cp`.

### Orchestration (Dagster)

```powershell
$env:DAGSTER_HOME = "$PWD\data\dagster_home"   # persists run history
dagster dev                                     # UI at http://localhost:3000
```

```
gleif_raw ───────────────────────────┐
sanctions_raw ───────────────────────┤
sec_filings ─► supplier_extraction ──┼─► warehouse ─[quality_gate]─► entity_resolution ─[match_gate]─► risk_scores ─[risk_gate]─► neo4j_graph
news_raw ────────────────────────────┘
```

- **`quality_gate`** and **`match_gate`** are *blocking* asset checks: if any error-level check
  fails, nothing downstream is materialized, so bad data never reaches the graph.
- Set `use_llm: true` on the `sg` resource to let Claude adjudicate the review band.
- **Jobs:** `full_refresh` (weekly, Mon 05:00 UTC), `sanctions_refresh` (daily 06:00 UTC —
  sanctions change daily, ownership rarely, so only sanctions and its downstream assets rerun)
  and `news_refresh` (every 6 h).
- Set `extraction_method: llm` on the `sg` resource for Claude-extracted `SUPPLIES` links.
  Without `SEC_USER_AGENT`, the SEC assets report "skipped" instead of failing the run.
- Network assets retry twice with exponential backoff.
- Schedules start **stopped**; turn them on in the UI.

### Try it in Neo4j Browser

```cypher
// Ownership network around the seeds
MATCH p = (:Company)-[:OWNED_BY*1..2]->(:Company) RETURN p LIMIT 200;

// Companies linked to a sanctions record, with their corporate parents
MATCH (c:Company)-[m:SANCTIONED_AS]->(s:SanctionedEntity)
OPTIONAL MATCH (c)-[:OWNED_BY]->(parent:Company)
RETURN c.legal_name, s.name, s.program_ids, m.match_method, parent.legal_name;

// Subsidiaries whose parent is sanctioned (exposure by ownership)
MATCH (child:Company)-[:OWNED_BY]->(parent:Company)-[:SANCTIONED_AS]->(s)
RETURN child.legal_name, child.legal_country, parent.legal_name, s.program_ids;
```

## Entity resolution

Most sanctions records carry no identifier linking them to a company register, so SupplyGraph
matches on names and countries across scripts and legal-form conventions:

```powershell
supplygraph resolve            # fuzzy matching; borderline cases -> review queue
supplygraph resolve --llm      # Claude adjudicates the review band (needs Anthropic credentials)
supplygraph evaluate-er        # precision/recall on the committed gold set
```

| Test split (125 sanctioned, 318 clean companies) | Precision | Recall | False alarms |
|---|---:|---:|---:|
| Production (auto-match) | **0.969** | **0.760** | **0.63 %** |

Labels come from identifiers (LEI or registration number), never from names; thresholds are
tuned on a separate dev split. Ablations, the review-queue analysis, what went wrong in the first
evaluation, and known limitations are in [docs/entity-resolution.md](docs/entity-resolution.md).

```cypher
// Analyst review queue: candidates the matcher would not auto-approve
MATCH (c:Company)-[p:POSSIBLE_SANCTIONS_MATCH]->(s:SanctionedEntity)
RETURN c.legal_name, s.name, p.score, p.country_relation ORDER BY p.score DESC;
```

## Supply chains from filings, disruptions from news

Annual reports name key suppliers in prose. SupplyGraph pulls the latest 10-K / 20-F from SEC
EDGAR, finds the passages that mention supply relationships, extracts suppliers, resolves each
one to its legal entity in GLEIF, and links them into the graph with the verbatim sentence as
evidence.

```powershell
supplygraph extract-suppliers --method rules   # baseline -> POSSIBLE_SUPPLIER (review queue)
supplygraph extract-suppliers --method llm     # Claude -> SUPPLIES (needs Anthropic credentials)
supplygraph evaluate-extraction [--llm]        # precision/recall on the hand-labelled gold set
```

| Gold set: 82 passages, 12 filers, 27 supplier relationships | Precision | Recall |
|---|---:|---:|
| Rule baseline | 0.559 | 0.704 |
| Claude extractor | not yet measured (needs credentials) | |

The baseline can't tell a supplier from a competitor, customer or joint-venture partner, so its
links are leads, not facts. That is exactly the gap the Claude extractor is for. Why, how it's
measured, the entity-linking traps (CATL's German subsidiary, Samsung's Greek one), and the
GDELT relevance analysis are in [docs/extraction.md](docs/extraction.md).

```cypher
// Concentration risk: suppliers that several monitored companies depend on
MATCH (s:Company)-[:SUPPLIES|POSSIBLE_SUPPLIER]->(c:Company)
WITH s, collect(c.legal_name) AS customers WHERE size(customers) > 1
RETURN s.legal_name, s.legal_country, size(customers) AS n, customers ORDER BY n DESC;

// The evidence behind a link, straight from the filing
MATCH (s:Company)-[l:POSSIBLE_SUPPLIER]->(c:Company {legal_name: 'NVIDIA CORPORATION'})
RETURN s.legal_name, l.evidence, l.form, l.accession;

// Disruption signals on companies we monitor
MATCH (c:Company {watchlist: true})-[:MENTIONED_IN]->(n:NewsEvent)
RETURN c.legal_name, n.event_types, n.title, n.url ORDER BY n.published_at DESC;
```

On the live data, TSMC is a supplier to 6 of the 12 filers and Samsung Electronics to 3.

## Explainable risk scores and scenarios

Risk signals (sanctions matches, review-queue candidates, disruption news) propagate over
ownership and supplier links. Sanctions follow legal control; disruptions follow physical
dependency. Every score stores the path that produced it.

```powershell
supplygraph score-risk          # after resolve; scores land on Company nodes
supplygraph scenario TSMC       # what if TSMC is disrupted? who is hit, and through which links
supplygraph evaluate-risk       # weight sensitivity + supplier link-prediction baselines
```

```
Disruption of 台灣積體電路製造股份有限公司 (549300KB6NK5SBD14S87), severity 1.0: 6 companies affected
  +0.250  ADVANCED MICRO DEVICES, INC.
         via 台灣積體電路製造股份有限公司 -> ADVANCED MICRO DEVICES, INC. [POSSIBLE_SUPPLIER]
  ...
```

```cypher
// Highest-risk companies and why
MATCH (c:Company) WHERE c.risk_score > 0
RETURN c.legal_name, c.risk_score, c.inherited_risk, c.risk_drivers ORDER BY c.risk_score DESC;
```

Rosneft Deutschland GmbH scores 0.81 purely through ownership by its sanctioned parent. The ranking
holds up when weights change: every weight scaled ×0.5 or ×1.5 still keeps at least 9 of the top 10.
For supplier recommendation, a popularity baseline beats collaborative filtering on today's 30
links, which is why there is no GNN yet. Details and the decision record are in [docs/risk.md](docs/risk.md).

## Ask the graph; act through audited proposals

A Claude agent answers questions using read-only graph tools. Every fact it states cites a graph
ref (`[lei:…]`, `[sanction:…]`, `[filing:…]`, `[news:…]`) that a tool actually returned, and the
grounding is checked in code. It can *propose* the ontology's four actions, but only a human can
approve one. Needs Claude credentials (`ANTHROPIC_API_KEY` or `ant auth login`).

```powershell
supplygraph ask "Why is Rosneft Deutschland GmbH's risk score so high?"
supplygraph ask "Dismiss the ASML Hong Kong sanctions candidate - it only matched the alias 'Hong Kong'"
supplygraph actions list --status proposed
supplygraph actions approve A-470e5c7cd3 --by <analyst>    # applied at the next build / resolve
supplygraph evaluate-agent        # 22-case eval suite on a frozen fixture (grounding, actions, injection)
```

On the live data, approving that dismissal removes the spurious possible-sanction risk from
the whole ASML group (companies at risk: 24 → 7). Design, safety properties and the eval
suite are in [docs/agent.md](docs/agent.md).

## Explorer UI

A React + Cytoscape front end over a small read-only API (FastAPI, reusing the agent's graph
tools, so the UI and the agent see the same data through the same queries).

```powershell
cd web; npm install; npm run build; cd ..
supplygraph serve                 # http://127.0.0.1:8000  (API docs at /docs)
```

- **Dashboard:** key figures, a risk-distribution histogram, the highest-risk companies with
  their main driver, and the sanctions review queue.
- **Explorer:** search by name, short name or LEI, then see the ownership and supplier
  neighbourhood (1–3 hops). Node color is the risk score; sanctioned companies and review
  candidates get a status ring; unverified `POSSIBLE_SUPPLIER` links are dashed. The side panel
  shows risk drivers with their paths, sanctions matches, links with evidence sentences and
  filing refs, and news. The **Disruption scenario** view draws who is hit, and how.
- **Actions:** the agent's proposals with their evidence and the question that led to them.
  Approve, or reject with a note; this writes to the same audit log as the CLI.
- **Ask:** the Q&A agent with clickable citations. Needs Claude API credit; everything else
  works without it.

The server binds to localhost and has no authentication: it is a single-analyst tool. For
frontend development, `npm run dev` in `web/` proxies `/api` to a running `supplygraph serve`.

## Deploy and operate

```powershell
docker build -t supplygraph:0.8.0 .
helm install sg deploy/helm/supplygraph -n supplygraph --create-namespace --set api.token.existingSecret=<secret>
helm test sg -n supplygraph
```

- **One image for both workloads.** It runs the server (a Deployment with liveness and
  readiness probes) and the nightly pipeline (a CronJob).
- **Atomic publishing.** The pipeline builds the warehouse in a work directory and publishes it
  by atomic rename, and only if the quality gate passed, so the explorer never serves a broken
  or half-built warehouse.
- **Usable on install.** A fresh install serves the committed demo snapshot until the first run.
- **Monitoring.** `/metrics` exposes data-freshness, volume and traffic metrics, and the
  optional PrometheusRule alerts when risk scores go stale. Logs are JSON.
- **Write protection.** Writes need an API token; without one, a deployed server is read-only.

Verified on a kind cluster: install, `helm test`, token auth, and one real pipeline run. See
[deploy/README.md](deploy/README.md) for the guide, [docs/adr/](docs/adr/) for the twelve
architecture decision records, and [docs/demo.md](docs/demo.md) for a five-minute walkthrough.

## Development

```powershell
pytest --cov=supplygraph     # unit + Dagster tests, no network needed
ruff check .                 # lint

cd web; npm test; npm run build  # UI unit tests, typecheck, production build

# Neo4j integration tests (needs `docker compose up -d`; only touches TESTSG*/TEST-* nodes)
$env:SUPPLYGRAPH_INTEGRATION = "1"; pytest
```

CI (GitHub Actions) runs lint, Dagster definition validation, and all tests against a real
Neo4j service container on every push and PR.

## Roadmap

- [x] **M1 — Ingestion & graph:** GLEIF + OpenSanctions → DuckDB → quality checks → Neo4j
- [x] **M2 — Orchestration:** Dagster assets, blocking quality gate, schedules, CI with Neo4j
- [x] **M3 — Entity resolution:** registration-number links, fuzzy name/country matching, Claude adjudication, review queue, dev/test evaluation
- [x] **M4 — Filings & news:** supplier extraction from SEC 10-K/20-F (rules + Claude), GLEIF entity linking, GDELT disruption events, extraction eval
- [x] **M5 — Graph risk:** explainable risk propagation, disruption scenarios, weight sensitivity, supplier link-prediction benchmark (GNN deferred, see docs/risk.md)
- [x] **M6 — Agent:** GraphRAG Q&A over graph tools with grounded citations, proposal-only audited actions applied by the pipeline, 22-case eval suite (built and tested; live eval score pending API credit, see docs/agent.md)
- [x] **M7 — UI:** React + Cytoscape graph explorer, risk dashboard, scenario view, analyst decisions on agent proposals (`supplygraph serve`)
- [x] **M8 — Deploy:** container image, Helm chart (API + pipeline CronJob with atomic publish), API token auth, health/readiness probes, Prometheus metrics and alerts, JSON logs, 12 ADRs, demo walkthrough

## Data & licensing

- **GLEIF** LEI data — CC0.
- **OpenSanctions** — CC BY-NC 4.0: free for non-commercial use; commercial use requires a license.
- **SEC EDGAR** — public filings; follow SEC fair-access rules (≤ 10 req/s, User-Agent with a
  contact email via `SEC_USER_AGENT`).
- **GDELT** — free for unrestricted use with citation of the GDELT Project.

Code: MIT.
