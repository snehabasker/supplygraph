# SupplyGraph

**Supplier risk intelligence on an ontology-backed knowledge graph.**

> *"Which of our suppliers are exposed to sanctions, hidden ownership, or disruption — and what should we do about it?"*

SupplyGraph ingests open corporate-ownership and sanctions data, maps it into a typed ontology
(objects, links, actions), loads it into Neo4j, and (in later milestones) layers graph ML and a
GraphRAG agent on top.

## Architecture

```
GLEIF API ──┐                                 ┌─► Neo4j knowledge graph ─► GraphRAG agent (M6)
            ├─► raw JSONL/CSV ─► DuckDB ─► DQ ┤                          └─► Explorer UI (M7)
OpenSanctions┘     (bronze)      (silver) checks└─► Graph ML risk scores (M5)
```

| Layer | Tech | Why |
|---|---|---|
| Ingestion | `httpx`, retry + rate limiting | GLEIF allows 60 req/min; transient failures are retried, 404s are data |
| Warehouse | DuckDB | Zero-ops columnar SQL; full-refresh builds are idempotent |
| Quality | SQL checks, error/warn severity | Pipeline refuses to load the graph on blocking failures |
| Graph | Neo4j 5 | Native traversal for ownership chains; uniqueness constraints |

See [docs/ontology.md](docs/ontology.md) for the object/link model.

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env          # then set NEO4J_PASSWORD

docker compose up -d            # Neo4j at http://localhost:7474

supplygraph ingest-gleif        # ~2-3 min (rate-limited)
supplygraph ingest-sanctions    # ~80 MB download
supplygraph build
supplygraph check
supplygraph load-graph
# or: supplygraph run-all
```

On macOS/Linux use `source .venv/bin/activate` and `cp`.

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

## Development

```powershell
pytest --cov=supplygraph     # unit tests, no network needed
ruff check .                 # lint
```

## Roadmap

- [x] **M1 — Ingestion & graph:** GLEIF + OpenSanctions → DuckDB → quality checks → Neo4j
- [ ] **M2 — Orchestration:** Dagster assets, schedules, CI (GitHub Actions)
- [ ] **M3 — Entity resolution:** fuzzy name/country matching + LLM adjudication, precision/recall eval
- [ ] **M4 — Filings & news:** LLM extraction of suppliers from SEC 10-Ks; GDELT disruption events
- [ ] **M5 — Graph ML:** risk propagation, link prediction (PyTorch Geometric)
- [ ] **M6 — Agent:** GraphRAG Q&A + audited actions, evaluated with an eval suite
- [ ] **M7 — UI:** React + Cytoscape graph explorer and risk dashboard
- [ ] **M8 — Deploy:** Helm, observability, ADRs, demo

## Data & licensing

- **GLEIF** LEI data — CC0.
- **OpenSanctions** — CC BY-NC 4.0: free for non-commercial use; commercial use requires a license.

Code: MIT.
