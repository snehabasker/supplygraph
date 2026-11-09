"""Explorer API (M7): read endpoints on the committed real-data fixture, analyst decisions."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from supplygraph.agent.actions import DISMISS_MATCH, ActionStore
from supplygraph.agent.evaluate import load_fixture
from supplygraph.api import views
from supplygraph.api.app import create_app

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent"
TSMC = "549300KB6NK5SBD14S87"
NVIDIA = "549300S4KLFTLO7GSQ80"
ROSNEFT = "253400JT3MQWNDKMJE44"
ASML = "724500Y6DUVHQD6OXN27"
ASML_HK = "54930067Q5KFMVMCUS63"
ASML_HK_SANCTION = "NK-Y2FwAXzabJzeQc2EPzZ9iV"


@pytest.fixture(scope="module")
def graph():
    con, now = load_fixture(EVAL_DIR / "fixture")
    yield con, now
    con.close()


@pytest.fixture
def store(tmp_path) -> ActionStore:
    return ActionStore(tmp_path / "events.jsonl")


@pytest.fixture
def client(graph, store, tmp_path) -> TestClient:
    con, now = graph
    # A cursor is a separate connection to the same in-memory database, closed per request.
    app = create_app(con.cursor, store.path, now=now)
    return TestClient(app)


def test_summary_counts_match_the_fixture(client) -> None:
    s = client.get("/api/summary").json()
    assert s["companies"] == 278 and s["sanctioned"] == 1 and s["review_queue"] == 2
    assert s["supply_links"] == {"SUPPLIES": 0, "POSSIBLE_SUPPLIER": 30}
    assert sum(b["companies"] for b in s["risk_histogram"]) == s["at_risk"] == 24
    assert s["zero_risk"] + s["at_risk"] == s["companies"]
    assert s["proposed_actions"] == 0


def test_company_search_and_detail(client) -> None:
    hit = client.get("/api/search", params={"q": "TSMC"}).json()["matches"][0]
    assert hit["ref"] == f"lei:{TSMC}"
    detail = client.get(f"/api/companies/{ROSNEFT}").json()
    assert detail["sanctions_matches"][0]["match_method"] == "LEI_IDENTIFIER"
    rel = client.get(f"/api/companies/{NVIDIA}/relationships/suppliers").json()
    assert rel["total"] == 7


@pytest.mark.parametrize(("url", "code"), [
    ("/api/companies/NOPE", 404),
    ("/api/companies/NOPE/graph", 404),
    (f"/api/companies/{TSMC}/relationships/friends", 400),
    (f"/api/companies/{TSMC}/graph?depth=9", 422),
    (f"/api/companies/{TSMC}/scenario?severity=2", 422),
    ("/api/sanctions/NOPE", 404),
    ("/api/actions?status=maybe", 400),
    ("/api/search?q=", 422),
])
def test_bad_requests_are_rejected_with_the_right_status(client, url, code) -> None:
    assert client.get(url).status_code == code


def test_neighborhood_is_cytoscape_ready_and_consistent(graph) -> None:
    g = views.neighborhood(graph[0], TSMC, depth=1)
    ids = {n["id"] for n in g["nodes"]}
    assert TSMC in ids and NVIDIA in ids and not g["truncated"]
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
    assert len({e["id"] for e in g["edges"]}) == len(g["edges"])
    to_nvidia = next(e for e in g["edges"] if e["source"] == TSMC and e["target"] == NVIDIA)
    assert to_nvidia["type"] == "POSSIBLE_SUPPLIER" and to_nvidia["accession"]
    rosneft = views.neighborhood(graph[0], ROSNEFT)
    assert next(n for n in rosneft["nodes"] if n["id"] == ROSNEFT)["sanctioned"]


def test_neighborhood_is_capped(graph) -> None:
    g = views.neighborhood(graph[0], ASML, depth=3, max_nodes=5)
    assert len(g["nodes"]) == 5 and g["truncated"]
    assert {n["hops"] for n in g["nodes"]} <= {0, 1, 2, 3}


def test_scenario_graph_draws_each_impact_path(client) -> None:
    s = client.get(f"/api/companies/{TSMC}/scenario").json()
    impacted = {r["ref"][4:] for r in s["impacted"]}
    assert NVIDIA in impacted and len(impacted) == 6
    nodes = {n["id"]: n for n in s["nodes"]}
    assert nodes[TSMC]["shocked"] and nodes[NVIDIA]["delta"] > 0
    for r in s["impacted"]:  # every hop of every path is an edge
        p = r["path_leis"]
        for src, dst in pairwise(p):
            assert f"risk:{src}->{dst}" in {e["id"] for e in s["edges"]}


def test_analyst_decides_a_proposal_through_the_api(client, store, graph) -> None:
    a, _ = store.propose(graph[0], action=DISMISS_MATCH, lei=ASML_HK,
                         sanction_id=ASML_HK_SANCTION, reason="alias-only match",
                         evidence=[f"lei:{ASML_HK}"], proposed_by="agent:test")
    listed = client.get("/api/actions", params={"status": "proposed"}).json()["actions"]
    assert [x["action_id"] for x in listed] == [a.action_id]
    assert client.get("/api/summary").json()["proposed_actions"] == 1

    r = client.post(f"/api/actions/{a.action_id}/decision",
                    json={"approve": True, "by": "ana", "note": "checked"})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    again = client.post(f"/api/actions/{a.action_id}/decision",
                        json={"approve": False, "by": "bob"})
    assert again.status_code == 409
    assert client.post("/api/actions/A-nope/decision",
                       json={"approve": True, "by": "ana"}).status_code == 404
    assert client.post(f"/api/actions/{a.action_id}/decision",
                       json={"approve": True, "by": ""}).status_code == 422
    # The company profile shows the live action state.
    actions = client.get(f"/api/companies/{ASML_HK}").json()["analyst_actions"]
    assert actions[0]["status"] == "approved"


def test_ask_without_credentials_is_a_clear_503(client, monkeypatch) -> None:
    import anthropic

    class NoCredentials:
        beta = type("B", (), {"messages": type("M", (), {"create": staticmethod(
            lambda **_: (_ for _ in ()).throw(
                TypeError('"Could not resolve authentication method."')))})})

    monkeypatch.setattr(anthropic, "Anthropic", NoCredentials)
    r = client.post("/api/ask", json={"question": "Who supplies NVIDIA?"})
    assert r.status_code == 503 and "credentials" in r.json()["detail"]


def test_built_ui_is_served_with_spa_fallback(graph, store, tmp_path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    c = TestClient(create_app(graph[0].cursor, store.path, static_dir=dist))
    assert c.get("/").text == "<html>app</html>"
    assert c.get("/company/XYZ").text == "<html>app</html>"  # client-side route
    assert c.get("/assets/app.js").text == "console.log(1)"
    assert "nope" not in c.get("/../secret.txt").text  # no escaping the static dir
    assert c.get("/api/unknown").status_code == 404
    assert c.get("/api/summary").status_code == 200
