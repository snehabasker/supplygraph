"""Celery background jobs (supplygraph/worker.py).

Unit tests run tasks in-process with `Task.apply()`: no broker and no network. The
integration test (SUPPLYGRAPH_INTEGRATION=1 and a Redis at CELERY_BROKER_URL, e.g.
`docker compose up -d --wait redis`) sends a job through a real Redis to an in-process worker.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

pytest.importorskip("celery")

from supplygraph import cli, pipeline, worker
from supplygraph.config import Settings
from supplygraph.er.vectors import ExactIndex


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace the pipeline functions the tasks wrap; record each call."""
    log: list = []

    def fake_resolve(settings, use_llm=False):
        log.append(("resolve", use_llm))
        return {"matches": 3, "review": 1}

    def fake_index(settings):
        log.append(("index",))
        return {"sanction_records": 2, "indexed_names": 3, "rebuilt": 1}

    monkeypatch.setattr(pipeline, "resolve_entities", fake_resolve)
    monkeypatch.setattr(pipeline, "index_er_vectors", fake_index)
    return log


def test_tasks_are_thin_wrappers_over_the_pipeline(calls: list) -> None:
    assert worker.resolve_entities.apply(kwargs={"use_llm": True}).get() == {"matches": 3,
                                                                             "review": 1}
    assert worker.index_er_vectors.apply().get()["rebuilt"] == 1
    assert calls == [("resolve", True), ("index",)]


def test_each_job_is_routed_to_its_queue() -> None:
    router = worker.app.amqp.router
    for task, queue in [(worker.TASK_RESOLVE, "warehouse"), (worker.TASK_INDEX_VECTORS, "vectors")]:
        assert router.route({}, task)["queue"].name == queue


def test_reliability_settings() -> None:
    conf = worker.app.conf
    assert conf.task_acks_late and conf.task_reject_on_worker_lost  # redeliver, never drop
    assert conf.worker_prefetch_multiplier == 1
    assert conf.accept_content == ["json"]  # no pickle from the broker


def test_warehouse_lock_conflict_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []

    def flaky(settings, use_llm=False):
        attempts.append(1)
        if len(attempts) < 3:
            raise duckdb.IOException('Could not set lock on file "warehouse.duckdb"')
        return {"matches": 5}

    monkeypatch.setattr(pipeline, "resolve_entities", flaky)
    result = worker.resolve_entities.apply()
    assert result.successful() and result.get() == {"matches": 5}
    assert len(attempts) == 3


def test_non_transient_errors_fail_fast_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []

    def broken(settings):
        attempts.append(1)
        raise ValueError("indexing needs PGVECTOR_DSN")

    monkeypatch.setattr(pipeline, "index_er_vectors", broken)
    result = worker.index_er_vectors.apply()
    assert result.failed() and isinstance(result.result, ValueError)
    assert len(attempts) == 1


def test_warehouse_queue_refuses_concurrency_above_one() -> None:
    with pytest.raises(ValueError, match="concurrency 1"):
        worker.worker_argv(["warehouse", "vectors"], concurrency=2)
    argv = worker.worker_argv(["vectors"], concurrency=4, pool="threads")
    assert argv[argv.index("-Q") + 1] == "vectors" and argv[argv.index("-c") + 1] == "4"
    with pytest.raises(ValueError, match="unknown queue"):
        worker.worker_argv(["default"], concurrency=1)


def test_default_pool_is_solo_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.sys, "platform", "win32")
    argv = worker.worker_argv(["warehouse"], 1)
    assert argv[argv.index("-P") + 1] == "solo"


def test_unknown_job_is_rejected_before_reaching_the_broker() -> None:
    with pytest.raises(ValueError, match="unknown job"):
        worker.submit("delete-everything")


def test_cli_worker_rejects_bad_concurrency_without_starting() -> None:
    result = CliRunner().invoke(cli.app, ["worker", "--queues", "warehouse", "--concurrency", "3"])
    assert result.exit_code == 2 and "concurrency 1" in result.output


# --------------------------------------------------------------- the embedding pipeline step

class FakePgVectorIndex(ExactIndex):
    """ExactIndex with PgVectorIndex's context-manager and `rebuilt` interface."""

    builds = 0

    def __init__(self, dsn: str) -> None:
        super().__init__()
        self.rebuilt = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def build(self, rows):
        super().build(rows)
        FakePgVectorIndex.builds += 1
        self.rebuilt = True


def _settings(tmp_path: Path, dsn: str | None) -> Settings:
    return Settings(data_dir=tmp_path, neo4j_uri="bolt://x", neo4j_user="n", neo4j_password="p",
                    opensanctions_url="http://x", pgvector_dsn=dsn)


def test_index_er_vectors_embeds_org_sanctions_names(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    import supplygraph.er.vectors as vectors

    monkeypatch.setattr(vectors, "PgVectorIndex", FakePgVectorIndex)
    con = duckdb.connect(str(tmp_path / "warehouse.duckdb"))
    con.execute("CREATE TABLE sanctions (id VARCHAR, schema VARCHAR, name VARCHAR, "
                "aliases VARCHAR, countries VARCHAR)")
    con.execute("INSERT INTO sanctions VALUES "
                "('s1', 'Company', 'Gazprom Neft PJSC', 'Gazpromneft', 'ru'), "
                "('s2', 'Organization', 'Rosatom', NULL, 'ru'), "
                "('s3', 'Person', 'Ivan Petrov', NULL, 'ru')")  # people are not ER targets
    con.close()

    out = pipeline.index_er_vectors(_settings(tmp_path, "postgresql://fake"))
    assert out == {"sanction_records": 2, "indexed_names": 3, "rebuilt": 1}


def test_index_er_vectors_requires_a_dsn(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PGVECTOR_DSN"):
        pipeline.index_er_vectors(_settings(tmp_path, None))


# --------------------------------------------------------------- real Redis round trip

@pytest.mark.skipif(os.getenv("SUPPLYGRAPH_INTEGRATION") != "1",
                    reason="set SUPPLYGRAPH_INTEGRATION=1 with Redis at CELERY_BROKER_URL")
def test_job_round_trip_through_redis(calls: list) -> None:
    from celery.contrib.testing.worker import start_worker

    with start_worker(worker.app, pool="solo", queues=["warehouse", "vectors"],
                      perform_ping_check=False, shutdown_timeout=10):
        res = worker.submit("resolve", use_llm=False)
        assert res.get(timeout=30) == {"matches": 3, "review": 1}
        assert worker.status(res.id)["state"] == "SUCCESS"
    assert calls == [("resolve", False)]
