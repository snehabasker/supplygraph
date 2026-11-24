"""Background jobs with Celery (Redis broker + result backend).

Dagster still runs the scheduled batch pipeline (orchestration/). Celery is for on-demand
work that should not block the caller: re-resolving entities after an analyst decision, or
re-embedding sanctions names into pgvector after a sanctions refresh. See docs/adr/0015.

Every task is a thin wrapper over `pipeline.py` (no duplicated logic). The design follows
from two properties of the pipeline:

* Steps are idempotent, so a task can safely run twice. Tasks are acknowledged *after* they
  finish (`acks_late`), and a job lost with a crashed worker is redelivered, not dropped.
* DuckDB allows one writing process per file. Jobs that write the warehouse go to the
  `warehouse` queue, served by a single-slot worker (`supplygraph worker` refuses
  concurrency > 1 for it). A lock conflict with another process (a CLI run, Dagster) raises
  `duckdb.IOException`, which is retried with exponential backoff instead of failing.

Queues:
    warehouse   resolve_entities        writes er_* tables, concurrency 1
    vectors     index_er_vectors        reads the warehouse, writes pgvector
"""

from __future__ import annotations

import sys

import duckdb
from celery import Celery
from celery.result import AsyncResult

from supplygraph import pipeline
from supplygraph.config import Settings, load_settings

QUEUE_WAREHOUSE = "warehouse"
QUEUE_VECTORS = "vectors"

TASK_RESOLVE = "supplygraph.resolve_entities"
TASK_INDEX_VECTORS = "supplygraph.index_er_vectors"
TASK_ROUTES = {
    TASK_RESOLVE: {"queue": QUEUE_WAREHOUSE},
    TASK_INDEX_VECTORS: {"queue": QUEUE_VECTORS},
}


def _transient_errors() -> tuple[type[BaseException], ...]:
    """Errors worth retrying: the warehouse is locked by another process, or Postgres or the
    network is briefly unavailable. Anything else (bad data, a bug) fails fast."""
    errors: list[type[BaseException]] = [duckdb.IOException, ConnectionError, TimeoutError]
    try:
        import psycopg

        errors.append(psycopg.OperationalError)
    except ImportError:  # the `vectors` extra is optional
        pass
    return tuple(errors)


TRANSIENT = _transient_errors()
RETRY = {"autoretry_for": TRANSIENT, "retry_backoff": 5, "retry_backoff_max": 300,
         "retry_jitter": True, "max_retries": 8}


def make_app(settings: Settings | None = None) -> Celery:
    settings = settings or load_settings()
    app = Celery("supplygraph", broker=settings.celery_broker_url,
                 backend=settings.celery_result_backend or settings.celery_broker_url)
    app.conf.update(
        task_serializer="json", result_serializer="json", accept_content=["json"],
        task_routes=TASK_ROUTES,
        task_acks_late=True,  # idempotent tasks: redeliver rather than lose
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,  # long jobs: don't hoard messages on one worker
        task_track_started=True,  # status shows STARTED, not just PENDING
        task_soft_time_limit=30 * 60,
        task_time_limit=35 * 60,
        result_expires=7 * 24 * 3600,
        broker_connection_retry_on_startup=True,
    )
    return app


app = make_app()


@app.task(name=TASK_RESOLVE, **RETRY)
def resolve_entities(use_llm: bool = False) -> dict[str, int]:
    return pipeline.resolve_entities(load_settings(), use_llm=use_llm)


@app.task(name=TASK_INDEX_VECTORS, **RETRY)
def index_er_vectors() -> dict[str, int]:
    return pipeline.index_er_vectors(load_settings())


JOBS = {"resolve": resolve_entities, "index-vectors": index_er_vectors}


def submit(job: str, **kwargs) -> AsyncResult:
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; choose from {sorted(JOBS)}")
    return JOBS[job].apply_async(kwargs=kwargs)


def status(task_id: str) -> dict:
    res = AsyncResult(task_id, app=app)
    out: dict = {"id": task_id, "state": res.state}
    if res.successful():
        out["result"] = res.result
    elif res.failed():
        out["error"] = repr(res.result)
    return out


def worker_argv(queues: list[str], concurrency: int, pool: str | None = None) -> list[str]:
    """Arguments for `celery worker`. Refuses to serve the warehouse queue with more than
    one slot: two concurrent writers would only fight over the DuckDB file lock."""
    unknown = set(queues) - {QUEUE_WAREHOUSE, QUEUE_VECTORS}
    if unknown:
        raise ValueError(f"unknown queue(s) {sorted(unknown)}")
    if QUEUE_WAREHOUSE in queues and concurrency > 1:
        raise ValueError("the warehouse queue needs concurrency 1 (DuckDB has a single writer); "
                         "run a separate worker for the vectors queue")
    # prefork is not supported on Windows; solo runs one task at a time in the main process
    pool = pool or ("solo" if sys.platform == "win32" else "prefork")
    return ["worker", "-Q", ",".join(queues), "-c", str(concurrency), "-P", pool,
            "-n", f"supplygraph-{'-'.join(queues)}@%h", "-l", "INFO"]
