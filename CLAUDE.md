# SupplyGraph — working agreement for Claude

## Commands (Windows; venv at .venv)
- Test: `.\.venv\Scripts\python.exe -m pytest`
- Lint: `.\.venv\Scripts\python.exe -m ruff check .`
- Pipeline: `.\.venv\Scripts\supplygraph.exe <ingest-gleif|ingest-sanctions|build|check|load-graph|run-all>`
- Neo4j: `docker compose up -d --wait`; queries via `docker exec -i supplygraph-neo4j cypher-shell` (pipe from bash, not PowerShell — PowerShell adds a BOM)

## Definition of done (every change)
1. Tests added/updated for the new behaviour, including at least one failure/edge case.
2. `pytest` and `ruff check .` both pass.
3. If the change touches ingestion, warehouse or graph: run it against real data and show the counts.
4. Every bug fixed gets a regression test that fails without the fix.

## Rules
- `docs/ontology.md` is the contract. Changing an object/link type means updating it first.
- All pipeline steps must be idempotent (CREATE OR REPLACE in DuckDB, MERGE in Neo4j).
- Unit tests never hit the network; use fakes / `httpx.MockTransport`.
- Values passed to Neo4j must be str/int/float/bool/None (DuckDB DECIMAL → Decimal breaks the driver).
- No secrets in code; config comes from `.env` (see `.env.example`).
- Commits: no Co-Authored-By Claude trailer.
