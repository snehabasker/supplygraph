"""Multi-hop Cypher tools (agent/cypher.py) and the MCP server, with a fake Neo4j driver.

The real queries run against Neo4j in test_neo4j_integration.py (opt-in)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

from supplygraph.agent import cypher as cy
from supplygraph.agent.agent import Agent
from supplygraph.agent.evaluate import load_fixture
from supplygraph.agent.tools import TOOL_SPECS, GraphTools, ToolError

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent"
ROOT_LEI = "549300S4KLFTLO7GSQ80"
SUP, SUP2 = "254900DIEONKUALBFM06", "30030004CM3GSZXX7O56"


class FakeTx:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver

    def run(self, query: str, **params):
        self.driver.queries.append((query, params))
        rows = self.driver.respond(query, params)
        return [SimpleNamespace(data=lambda r=r: r) for r in rows]


class FakeSession:
    def __init__(self, driver: FakeDriver, kwargs: dict) -> None:
        self.driver, self.kwargs = driver, kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute_read(self, work):
        # neo4j.unit_of_work(timeout=...) stores the server-side timeout on the function.
        self.driver.reads.append({**self.kwargs, "timeout": getattr(work, "timeout", None)})
        if self.driver.fail is not None:
            raise self.driver.fail
        return work(FakeTx(self.driver))


class FakeDriver:
    """Answers the root lookup for known LEIs and returns `rows` for every path query."""

    def __init__(self, rows: list[dict] | None = None, known=(ROOT_LEI,), fail=None) -> None:
        self.rows, self.known, self.fail = rows or [], set(known), fail
        self.queries: list[tuple[str, dict]] = []
        self.reads: list[dict] = []

    def session(self, **kwargs):
        return FakeSession(self, kwargs)

    def respond(self, query: str, params: dict) -> list[dict]:
        if query == cy.ROOT:
            return [{"name": "NVIDIA CORPORATION"}] if params["lei"] in self.known else []
        return self.rows


def hop(a: str, t: str, b: str, **kw) -> dict:
    return {"from": a, "type": t, "to": b, "relationship_type": kw.get("rtype"),
            "accession": kw.get("acc")}


@pytest.fixture(scope="module")
def fixture_con():
    con, now = load_fixture(EVAL_DIR / "fixture", EVAL_DIR / "planted.json")
    yield con, now
    con.close()


# --- query safety -------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["supplier_tree", "sanctions_exposure", "disruption_reach",
                                  "root"])
def test_tool_queries_are_read_only_for_every_depth(name) -> None:
    for depth in range(1, cy.MAX_DEPTH + 1):
        q = cy.QUERIES[name]
        if name != "root":
            q = q.format(depth=depth, types=cy.SUPPLY)
        assert not cy.WRITE_CLAUSES.search(q), q


def test_a_write_query_is_refused_before_it_reaches_the_database() -> None:
    driver = FakeDriver()
    with pytest.raises(ToolError, match="not read-only"):
        cy.CypherTools(driver).read("MATCH (c:Company) DETACH DELETE c")
    assert driver.queries == []


def test_queries_run_in_read_sessions_with_a_server_timeout() -> None:
    driver = FakeDriver()
    cy.CypherTools(driver, timeout=2.5).supplier_tree(ROOT_LEI)
    import neo4j

    assert driver.reads and all(r["default_access_mode"] == neo4j.READ_ACCESS
                                and r["timeout"] == 2.5 for r in driver.reads)


@pytest.mark.parametrize("depth", [0, cy.MAX_DEPTH + 1, True, "2", 2.0])
def test_depth_must_be_a_bounded_integer(depth) -> None:
    driver = FakeDriver()
    with pytest.raises(ToolError, match="depth"):
        cy.CypherTools(driver).supplier_tree(ROOT_LEI, depth=depth)
    assert driver.queries == []  # rejected before any query: depth is formatted into Cypher


def test_depth_and_link_types_are_formatted_into_the_path_pattern() -> None:
    driver = FakeDriver()
    tools = cy.CypherTools(driver)
    tools.supplier_tree(ROOT_LEI, depth=3)
    tools.supplier_tree(ROOT_LEI, depth=1, asserted_only=True)
    tools.sanctions_exposure(ROOT_LEI, max_hops=4)
    paths = [q for q, _ in driver.queries if q != cy.ROOT]
    assert "[:SUPPLIES|POSSIBLE_SUPPLIER*1..3]" in paths[0]
    assert "[:SUPPLIES*1..1]" in paths[1]
    assert "{0,4}" in paths[2]
    assert all(p["lei"] == ROOT_LEI for q, p in driver.queries)


def test_unknown_company_and_unavailable_database_are_tool_errors() -> None:
    from neo4j.exceptions import ServiceUnavailable

    with pytest.raises(ToolError, match="no company"):
        cy.CypherTools(FakeDriver(known=())).disruption_reach(ROOT_LEI)
    with pytest.raises(ToolError, match="unavailable"):
        cy.CypherTools(FakeDriver(fail=ServiceUnavailable("down"))).supplier_tree(ROOT_LEI)


def test_timeout_is_reported_as_a_tool_error() -> None:
    from neo4j.exceptions import ClientError

    class TimedOut(ClientError):
        code = "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration"

    with pytest.raises(ToolError, match="exceeded"):
        cy.CypherTools(FakeDriver(fail=TimedOut("too slow"))).supplier_tree(ROOT_LEI)


# --- result shaping -----------------------------------------------------------------------

def test_supplier_tree_marks_a_path_through_a_lead_as_a_lead() -> None:
    rows = [
        {"lei": SUP, "name": "Wistron", "country": "TW", "risk_score": 0.0, "tier": 1,
         "hops": [hop(SUP, "SUPPLIES", ROOT_LEI, acc="A-1")]},
        {"lei": SUP2, "name": "Hon Hai", "country": "TW", "risk_score": 0.1, "tier": 2,
         "hops": [hop(SUP2, "POSSIBLE_SUPPLIER", SUP, acc="A-2"),
                  hop(SUP, "SUPPLIES", ROOT_LEI, acc="A-1")]},
    ]
    out = cy.CypherTools(FakeDriver(rows)).supplier_tree(ROOT_LEI, depth=2)
    first, second = out["suppliers"]
    assert out["ref"] == f"lei:{ROOT_LEI}" and out["total"] == 2
    assert first["certainty"] == "asserted" and second["certainty"] == "lead"
    assert second["path"][0] == {"from_ref": f"lei:{SUP2}", "link": "POSSIBLE_SUPPLIER",
                                 "to_ref": f"lei:{SUP}", "certainty": "lead",
                                 "filing_ref": "filing:A-2"}


def test_sanctions_exposure_separates_findings_from_candidates() -> None:
    rows = [
        {"sanction_id": "S-1", "sanction_name": "Rosneft", "match_type": "SANCTIONED_AS",
         "match_method": "LEI_IDENTIFIER", "confidence": 1.0, "via_lei": SUP, "via_name": "R",
         "hops_to_company": 1, "hops": [hop(ROOT_LEI, "OWNED_BY", SUP, rtype="DIRECT")]},
        {"sanction_id": "S-2", "sanction_name": "RT-Capital",
         "match_type": "POSSIBLE_SANCTIONS_MATCH", "match_method": None, "confidence": 0.95,
         "via_lei": SUP2, "via_name": "RN", "hops_to_company": 2, "hops": []},
    ]
    out = cy.CypherTools(FakeDriver(rows)).sanctions_exposure(ROOT_LEI)
    assert out["exposed"] is True
    assert [(e["sanction_ref"], e["certainty"]) for e in out["exposures"]] == [
        ("sanction:S-1", "finding"), ("sanction:S-2", "candidate")]
    assert out["exposures"][0]["path"][0]["link"] == "OWNED_BY:DIRECT"


def test_no_exposure_is_an_explicit_false() -> None:
    out = cy.CypherTools(FakeDriver([])).sanctions_exposure(ROOT_LEI)
    assert out["exposed"] is False and out["exposures"] == []


# --- wiring into GraphTools and the agent -------------------------------------------------

def test_cypher_tools_are_offered_only_with_a_graph(fixture_con) -> None:
    con, now = fixture_con
    plain = GraphTools(con, now=now)
    assert plain.specs == TOOL_SPECS and "supplier_tree" not in plain.handlers
    with pytest.raises(ToolError, match="unknown tool"):
        plain.call("supplier_tree", {"lei": ROOT_LEI})

    graph = GraphTools(con, now=now, cypher=cy.CypherTools(FakeDriver()))
    assert [s["name"] for s in graph.specs][-3:] == sorted(cy.CYPHER_TOOLS, key=[
        "supplier_tree", "sanctions_exposure", "disruption_reach"].index)
    assert graph.call("supplier_tree", {"lei": ROOT_LEI})["total"] == 0


def test_agent_passes_cypher_tool_specs_to_the_model(fixture_con) -> None:
    con, now = fixture_con
    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="INSUFFICIENT DATA")],
                               stop_reason="end_turn", usage=None)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    Agent(con, client=client, now=now, model="fake",
          cypher=cy.CypherTools(FakeDriver())).ask("Who supplies NVIDIA's suppliers?")
    Agent(con, client=client, now=now, model="fake").ask("Who supplies NVIDIA?")
    with_graph, without = ({t["name"] for t in c["tools"]} for c in calls)
    assert with_graph >= cy.CYPHER_TOOLS and not cy.CYPHER_TOOLS & without


# --- MCP server ---------------------------------------------------------------------------

def run_mcp(tools: GraphTools, *calls: tuple[str, dict]):
    from mcp import Client

    from supplygraph.mcp_server import build_server

    async def go():
        async with Client(build_server(tools)) as client:
            listed = await client.list_tools()
            results = [await client.call_tool(name, args) for name, args in calls]
            return listed.tools, results

    return anyio.run(go)


def test_mcp_server_exposes_the_agents_read_only_tools(fixture_con) -> None:
    con, now = fixture_con
    tools = GraphTools(con, now=now, cypher=cy.CypherTools(FakeDriver()))
    listed, [found] = run_mcp(tools, ("search_companies", {"query": "NVIDIA"}))
    assert [t.name for t in listed] == [s["name"] for s in tools.specs]
    assert not any(t.name.startswith("propose_") for t in listed)  # no actions over MCP
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint
               for t in listed)
    assert not found.is_error
    assert found.structured_content["matches"][0]["ref"] == f"lei:{ROOT_LEI}"
    assert json.loads(found.content[0].text) == found.structured_content


def test_mcp_tool_errors_are_results_not_crashes(fixture_con) -> None:
    con, now = fixture_con
    _, results = run_mcp(GraphTools(con, now=now),
                         ("get_company", {"lei": "NOT-A-LEI"}),
                         ("supplier_tree", {"lei": ROOT_LEI}),  # no graph: not offered
                         ("get_news", {"lei": ROOT_LEI, "bogus": 1}))
    assert [r.is_error for r in results] == [True, True, True]
    assert "no company" in results[0].content[0].text
    assert "unknown tool" in results[1].content[0].text
