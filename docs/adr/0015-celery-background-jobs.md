# 0015. Celery + Redis for on-demand background jobs; Dagster stays the scheduler

**Status:** Accepted (M11)

## Context
Two jobs are slow enough that the caller shouldn't wait on them, and they are triggered by
events rather than by the clock:

- **Re-resolving entities** after an analyst approves a `confirm_match` / `dismiss_match`
  action ([0009](0009-proposal-only-actions-append-only-log.md)). Today the decision only
  takes effect at the next `resolve` run.
- **Re-embedding sanctions names** into pgvector ([0014](0014-pgvector-candidate-generation-opt-in.md))
  after a sanctions refresh. Measured on the real data, this takes about 2.5 minutes:
  25,336 organisations and 87,461 names, embedded in pure Python.

Dagster ([0004](0004-idempotent-pipeline-one-implementation.md)) schedules the batch
pipeline well, but it is built around asset materialisations, not a "do this one thing now"
queue that an API or CLI can call.

## Decision
- `supplygraph/worker.py` defines a Celery app, with Redis as both broker and result backend
  (`docker compose up -d --wait redis`). Tasks are one-line wrappers over `pipeline.py`, so
  there is still one implementation of every step.
- There are two queues, because DuckDB allows one writing process per file:
  - `warehouse` (`resolve_entities`) runs with concurrency 1. `supplygraph worker` refuses
    a higher value for this queue.
  - `vectors` (`index_er_vectors`) only reads the warehouse and writes Postgres.
- **Delivery is at-least-once**, which is safe because every step is idempotent: tasks are
  acknowledged after they finish (`acks_late`, `task_reject_on_worker_lost`), with prefetch 1.
- **Errors are classified.** A warehouse lock held by another process (`duckdb.IOException`),
  Postgres being down, or a network error is retried with exponential backoff and jitter,
  up to 8 attempts. Anything else fails at once with its error stored in the result backend.
- JSON serialisation only; pickle is never accepted from the broker.
- CLI: `supplygraph worker`, `supplygraph enqueue <resolve|index-vectors> [--wait]` and
  `supplygraph job <id>`.

## Verified on real data
| Check | Result |
|---|---|
| `enqueue index-vectors` | 25,336 records, 87,461 names; `rebuilt: 0`. The fingerprint matched the index that `evaluate-er --vectors` had built, so both build the same index. |
| `enqueue resolve` vs `supplygraph resolve` | identical counts (2 candidates, 2 to review) |
| Another process holding the warehouse write lock | the job went to RETRY (`IOException … Cannot open file`), then SUCCESS after the lock was released |
| Integration test | a job sent through a real Redis to an in-process worker |

## Rejected
- **Dagster sensors or run requests for on-demand work.** That's possible, but it turns a
  one-off job into a full run with run-storage overhead, and the API would need the Dagster
  GraphQL client to trigger it.
- **FastAPI `BackgroundTasks`.** These run inside the API process: lost on restart, no
  retries, no status after the request ends, and no way to keep concurrent warehouse writes
  to one.
- **RQ or Dramatiq.** Both are fine choices. Celery was chosen for its routing, the retry
  and late-ack options it ships with, and because it is the most common choice in Python
  teams.

## Consequences
- Redis is an optional service, like pgvector. Without it, everything still runs
  synchronously through the CLI and Dagster.
- **Next step, not done yet:** have the API enqueue `resolve` when an analyst decision is
  approved (behind `require_write`, [0010](0010-api-write-protection-modes.md)), and run the
  worker as a Deployment in the Helm chart.
- The embedding job's cost is in Python trigram hashing before the fingerprint check. If the
  cost matters, cache the fingerprint per sanctions snapshot and skip unchanged data first.
