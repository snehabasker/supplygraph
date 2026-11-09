# SupplyGraph — working agreement for Claude

## Commands (Windows; venv at .venv)
- Test: `.\.venv\Scripts\python.exe -m pytest`
- Lint: `.\.venv\Scripts\python.exe -m ruff check .`
- Pipeline: `.\.venv\Scripts\supplygraph.exe <ingest-gleif|ingest-sanctions|ingest-filings|extract-suppliers|ingest-news|build|resolve|score-risk|scenario|check|load-graph|run-all>`
- Integration tests: set `$env:SUPPLYGRAPH_INTEGRATION="1"` (Neo4j must be up); they only touch `TESTSG*` LEIs / `TEST-*` ids
- Dagster: `dagster definitions validate -m supplygraph.orchestration.definitions`; `dagster dev` for the UI
- Explorer UI: `cd web; npm test; npm run build` (typecheck + build), then `supplygraph serve`;
  API tests in `tests/test_api.py` run on the committed agent-eval fixture
- Pipeline logic lives in `pipeline.py`; CLI and Dagster assets are thin wrappers — never duplicate logic.
- Neo4j: `docker compose up -d --wait`; queries via `docker exec -i supplygraph-neo4j cypher-shell` (pipe from bash, not PowerShell — PowerShell adds a BOM)

## Definition of done (every change)
1. Tests added/updated for the new behaviour, including at least one failure/edge case.
2. `pytest` and `ruff check .` both pass.
3. If the change touches ingestion, warehouse or graph: run it against real data and show the counts.
4. Every bug fixed gets a regression test that fails without the fix.

## Entity resolution
- Any change to `er/` (normalization, scoring, thresholds) must be followed by `supplygraph evaluate-er`;
  report the test-split precision / recall / false-alarm deltas. Never tune thresholds on the test split.
- When metrics move, read the listed false alarms and misses before concluding — label noise is real
  (see docs/entity-resolution.md).
- Claude code: model from `SUPPLYGRAPH_LLM_MODEL` (default `claude-opus-5-5`); structured output via
  `client.beta.messages.parse`; failures go to review, never to a silent decision.

## Supplier extraction and news
- Any change to `extract/` must be followed by `supplygraph evaluate-extraction`; report the
  precision / recall deltas and read the listed false positives. `test_committed_eval_set...`
  pins the rule baseline's TP/FP/FN — update docs/extraction.md when it moves.
- Never edit `eval/extraction/gold.csv` to make a number go up; it records what the passage
  states. Rebuilding passages (`build-extraction-eval-set`) requires re-checking gold coverage.
- Link type follows method: LLM -> `SUPPLIES`, RULES -> `POSSIBLE_SUPPLIER` (enforced by a check).
- Name -> LEI resolution changes must bump `RESOLVER_VERSION` in `extract/link.py`.
- `SEC_USER_AGENT` lives only in `.env`; it contains a personal email — never commit it.

## Risk engine
- Weights live in `risk/engine.py:RiskConfig`; any change must be followed by
  `supplygraph evaluate-risk` and docs/risk.md updated (sensitivity + link-prediction tables).
- Keep the properties: bounded, monotone, cycle-safe, every non-zero score explained.
- Which link types a signal travels is defined in `SIGNAL_RELATIONS` — sanctions follow
  ownership, disruptions follow supply; don't let news flow parent -> subsidiary.
- A learned link-prediction model only ships if it beats the popularity baseline in
  `risk/linkpred.leave_one_out`.

## Agent (Q&A + actions)
- Any change to `agent/` (tools, prompt, loop) must be followed by `supplygraph evaluate-agent`
  (needs Claude credentials); report pass rate, grounding and unrequested-action deltas and read
  the failed answers. `evaluate-agent --check-only` (offline) must always pass.
- Never edit `eval/agent/cases.jsonl` to make a number go up; `build-agent-fixture` re-snapshots
  the fixture and requires fixing the stale cases it lists. `planted.json` holds synthetic rows.
- The agent only *proposes* actions; never add a tool that applies one. Approved actions are
  applied by `build` / `resolve` from `data/actions/events.jsonl` (append-only; never rewrite it).

## Explorer UI
- The API (`api/`) is read-only except analyst decisions, which go through `ActionStore.decide`
  (same audit log as the CLI). Company data comes from the agent's `GraphTools` — don't fork
  queries between UI and agent.
- Charts follow the data-viz rules: one sequential hue for risk, status colors only with a label,
  link type never by color alone. Look at a screenshot after UI changes.

## Deployment
- Image: `docker build -t supplygraph:0.8.0 .`; chart: `helm lint deploy/helm/supplygraph`.
  Test-deploy only to a dedicated kind cluster (`kind-supplygraph`), always with an explicit
  `--kube-context`; never to other projects' clusters.
- The API's write mode (open / token / off) is a security boundary (docs/adr/0010): new write
  endpoints must depend on `require_write`.
- Changing a recorded decision means writing a new ADR in `docs/adr/` that supersedes the old.

## Rules
- `docs/ontology.md` is the contract. Changing an object/link type means updating it first.
- All pipeline steps must be idempotent (CREATE OR REPLACE in DuckDB, MERGE in Neo4j).
- Unit tests never hit the network; use fakes / `httpx.MockTransport`.
- Values passed to Neo4j must be str/int/float/bool/None (DuckDB DECIMAL → Decimal breaks the driver).
- No secrets in code; config comes from `.env` (see `.env.example`).
- Commits: no Co-Authored-By Claude trailer.
