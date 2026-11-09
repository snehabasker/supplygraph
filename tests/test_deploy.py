"""Milestone 8: write protection, health/readiness, metrics, JSON logs, demo data, config."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from supplygraph import pipeline, quality, warehouse
from supplygraph.agent.actions import REQUEST_REVIEW, ActionStore
from supplygraph.agent.evaluate import load_fixture
from supplygraph.api.app import app_for_settings, create_app, write_mode
from supplygraph.config import Settings, load_settings
from supplygraph.logs import JsonFormatter

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent"
ROSNEFT_DE = "391200PKEUF3Y2NUMW25"
TOKEN = "s3cret-token"


@pytest.fixture(scope="module")
def graph():
    con, now = load_fixture(EVAL_DIR / "fixture")
    yield con, now
    con.close()


@pytest.fixture
def store(tmp_path) -> ActionStore:
    return ActionStore(tmp_path / "events.jsonl")


def make_client(graph, store, **kw) -> TestClient:
    return TestClient(create_app(graph[0].cursor, store.path, now=graph[1], **kw))


def proposal(graph, store) -> str:
    a, _ = store.propose(graph[0], action=REQUEST_REVIEW, lei=ROSNEFT_DE, sanction_id=None,
                         reason="r", evidence=[f"lei:{ROSNEFT_DE}"], proposed_by="agent:test")
    return a.action_id


# --- write protection ----------------------------------------------------------------------

@pytest.mark.parametrize(("host", "token", "mode"), [
    ("127.0.0.1", None, "open"), ("localhost", "", "open"), ("::1", None, "open"),
    ("0.0.0.0", None, "off"), ("10.0.0.5", None, "off"),
    ("0.0.0.0", TOKEN, "token"), ("127.0.0.1", TOKEN, "token"),
])
def test_write_mode(host, token, mode) -> None:
    assert write_mode(host, token) == mode


def test_token_mode_requires_the_bearer_token_for_writes_only(graph, store) -> None:
    c = make_client(graph, store, writes="token", token=TOKEN)
    action_id = proposal(graph, store)
    url, body = f"/api/actions/{action_id}/decision", {"approve": True, "by": "ana"}
    assert c.post(url, json=body).status_code == 401
    assert c.post(url, json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post(url, json=body, headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert c.post("/api/ask", json={"question": "q"}).status_code == 401
    assert store.get(action_id).status == "proposed"  # nothing was written
    ok = c.post(url, json=body, headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert c.get("/api/summary").status_code == 200  # reads stay open
    assert c.get("/api/config").json()["writes"] == "token"


def test_read_only_mode_refuses_every_write(graph, store) -> None:
    c = make_client(graph, store, writes="off")
    action_id = proposal(graph, store)
    r = c.post(f"/api/actions/{action_id}/decision", json={"approve": True, "by": "ana"},
               headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
    assert c.post("/api/ask", json={"question": "q"}).status_code == 403
    assert store.get(action_id).status == "proposed"
    assert c.get(f"/api/companies/{ROSNEFT_DE}").status_code == 200


def test_token_mode_without_a_token_is_a_configuration_error(graph, store) -> None:
    with pytest.raises(ValueError, match="needs a token"):
        create_app(graph[0].cursor, store.path, writes="token")


# --- health, readiness, metrics ------------------------------------------------------------

def test_health_and_readiness(graph, store, tmp_path) -> None:
    c = make_client(graph, store)
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/readyz").json() == {"status": "ready"}

    empty = TestClient(create_app(duckdb.connect(":memory:").cursor, store.path))
    r = empty.get("/readyz")
    assert r.status_code == 503 and r.json()["reason"] == "warehouse is empty"

    def broken():
        raise duckdb.IOException("Could not set lock on file")

    r = TestClient(create_app(broken, store.path)).get("/readyz")
    assert r.status_code == 503 and "lock" in r.json()["reason"]
    assert TestClient(create_app(broken, store.path)).get("/healthz").status_code == 200


def test_server_starts_before_the_first_pipeline_run(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", neo4j_uri="", neo4j_user="",
                        neo4j_password="", opensanctions_url="")
    c = TestClient(app_for_settings(settings))
    assert c.get("/healthz").status_code == 200
    assert c.get("/readyz").status_code == 503
    assert not settings.warehouse_path.exists()  # readiness probes don't create files


def metric(text: str, name: str, **labels) -> float | None:
    want = ",".join(f'{k}="{v}"' for k, v in labels.items())
    for line in text.splitlines():
        if line.startswith(f"{name}{{{want}}}" if labels else f"{name} "):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_metrics_expose_data_volume_freshness_and_traffic(graph, store) -> None:
    c = make_client(graph, store)
    proposal(graph, store)
    c.get(f"/api/companies/{ROSNEFT_DE}")
    c.get("/api/companies/NOPE")
    text = c.get("/metrics").text

    assert metric(text, "supplygraph_warehouse_up") == 1
    assert metric(text, "supplygraph_rows", kind="companies") == 278
    assert metric(text, "supplygraph_rows", kind="review_queue") == 2
    assert metric(text, "supplygraph_actions", status="proposed") == 1
    assert metric(text, "supplygraph_data_age_seconds", source="news") > 0
    route = "/api/companies/{lei}"
    assert metric(text, "supplygraph_http_requests_total", method="GET", route=route,
                  status="200") == 1
    assert metric(text, "supplygraph_http_requests_total", method="GET", route=route,
                  status="404") == 1
    assert ROSNEFT_DE not in text and "NOPE" not in text  # bounded label cardinality


def test_metrics_survive_an_unreadable_warehouse(store) -> None:
    def broken():
        raise duckdb.IOException("locked")

    text = TestClient(create_app(broken, store.path)).get("/metrics").text
    assert metric(text, "supplygraph_warehouse_up") == 0


# --- logs and config -----------------------------------------------------------------------

def test_json_log_lines_carry_extra_fields_and_exceptions() -> None:
    fmt = JsonFormatter()
    record = logging.LogRecord("supplygraph.x", logging.WARNING, __file__, 1, "hello %s",
                               ("world",), None)
    record.path = "/api/x"
    out = json.loads(fmt.format(record))
    assert out["message"] == "hello world" and out["level"] == "WARNING"
    assert out["logger"] == "supplygraph.x" and out["path"] == "/api/x"
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        err = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", None, sys.exc_info())
    assert "RuntimeError: boom" in json.loads(fmt.format(err))["exception"]


def test_actions_log_location_can_be_shared_across_data_dirs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SUPPLYGRAPH_DATA_DIR", str(tmp_path / "work"))
    monkeypatch.delenv("SUPPLYGRAPH_ACTIONS_LOG", raising=False)
    assert load_settings().actions_log == tmp_path / "work" / "actions" / "events.jsonl"
    monkeypatch.setenv("SUPPLYGRAPH_ACTIONS_LOG", str(tmp_path / "shared" / "events.jsonl"))
    assert load_settings().actions_log == tmp_path / "shared" / "events.jsonl"


# --- demo data -----------------------------------------------------------------------------

def test_demo_data_builds_a_valid_warehouse_and_never_overwrites_silently(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", neo4j_uri="", neo4j_user="",
                        neo4j_password="", opensanctions_url="")
    counts = pipeline.demo_data(settings, EVAL_DIR)
    assert counts["companies"] == 278 and counts["news_events"] == 14  # no planted rows
    with warehouse.connect(settings.warehouse_path) as con:
        assert not [r for r in quality.run_checks(con)
                    if r.status == "fail" and r.severity == quality.ERROR]
        assert con.execute("SELECT count(*) FROM news_events WHERE id LIKE 'TEST-%'"
                           ).fetchone()[0] == 0
        assert con.execute("SELECT scored_at FROM risk_config").fetchone()[0].year == 2026
        con.execute("DELETE FROM companies WHERE lei = ?", [ROSNEFT_DE])
    with pytest.raises(FileExistsError):
        pipeline.demo_data(settings, EVAL_DIR)
    pipeline.demo_data(settings, EVAL_DIR, force=True)
    with warehouse.connect(settings.warehouse_path) as con:
        assert con.execute("SELECT count(*) FROM companies").fetchone()[0] == 278
    assert not list(settings.data_dir.glob("*.demo-tmp"))
