# 0004. Idempotent full-refresh steps, one implementation, blocking quality gates

**Status:** Accepted (M1, M2)

## Context
The pipeline is re-run constantly: nightly, after a fix, from the CLI during development, and
from Dagster. Incremental updates would need change tracking on sources that don't provide it
(GLEIF relationships, the sanctions list).

## Decision
- **Every step is idempotent.** DuckDB tables are rebuilt with `CREATE OR REPLACE`; Neo4j
  writes use `MERGE`, plus pruning of links the warehouse no longer supports.
- **One implementation.** Pipeline logic lives in `pipeline.py`. The CLI and the Dagster
  assets are thin wrappers that never duplicate it.
- **Blocking gates.** Data-quality checks have `error` or `warn` severity. Any `error` failure
  stops the run before risk scoring or graph loading: in the CLI (`check` exits 1), in Dagster
  (blocking asset checks), and in Kubernetes, where a failed check means the warehouse is
  never published (0011).

## Alternatives considered
Incremental loads would be faster at large scale, but at ~300 companies and ~74 000 sanctions
records a full build takes about a minute, and full refresh can't accumulate drift.

## Consequences
- Re-running is always safe; a half-finished run is repaired by running again.
- Human decisions cannot live in rebuilt tables. They live in an append-only file the build
  reads (0009).
