"""Integration test against a real Neo4j.

Opt-in (SUPPLYGRAPH_INTEGRATION=1) because it needs a running database. It only touches
nodes with TESTSG* LEIs / TEST-* ids, so it is safe to run against a dev database.
"""

from __future__ import annotations

import os

import pytest

from conftest import make_lei, make_record, now_iso, write_raw_data
from supplygraph import graph, warehouse
from supplygraph.ingest import gleif

pytestmark = pytest.mark.skipif(
    os.getenv("SUPPLYGRAPH_INTEGRATION") != "1",
    reason="set SUPPLYGRAPH_INTEGRATION=1 with Neo4j running",
)

PARENT = make_lei("TESTSG000000000001")
CHILD = make_lei("TESTSG000000000002")
CLEAN = make_lei("TESTSG000000000003")

CLEANUP = """
MATCH (n) WHERE n.lei STARTS WITH 'TESTSG' OR n.id STARTS WITH 'TEST-'
DETACH DELETE n
"""


@pytest.fixture
def driver():
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "change-me-please")),
    )
    drv.verify_connectivity()
    drv.execute_query(CLEANUP)
    yield drv
    drv.execute_query(CLEANUP)
    drv.close()


@pytest.fixture
def con(tmp_path):
    records = {
        PARENT: make_record(PARENT, "Test Sanctioned Parent", "RU"),
        CHILD: make_record(CHILD, "Test Exposed Child", "DE"),
        CLEAN: make_record(CLEAN, "Test Clean Co", "US"),
    }
    sanctions = [{"id": "TEST-1", "schema": "Company", "name": "Test Sanctioned Parent",
                  "identifiers": PARENT, "program_ids": "TEST-PROGRAM", "last_seen": now_iso()}]
    write_raw_data(tmp_path, records, {(CHILD, PARENT, gleif.DIRECT)}, sanctions)
    c = warehouse.connect(":memory:")
    warehouse.build_all(c, tmp_path / "raw" / "gleif",
                        tmp_path / "raw" / "opensanctions" / "targets.simple.csv")
    yield c
    c.close()


def count(driver, query: str) -> int:
    records, _, _ = driver.execute_query(query)
    return records[0][0]


def test_load_is_idempotent_and_answers_exposure_query(driver, con) -> None:
    for _ in range(2):  # loading twice must not duplicate anything
        graph.load_graph(con, driver, batch_size=2)

    assert count(driver, "MATCH (c:Company) WHERE c.lei STARTS WITH 'TESTSG' "
                         "RETURN count(c)") == 3
    assert count(driver, "MATCH (:Company)-[r:OWNED_BY]->(p:Company) "
                         "WHERE p.lei STARTS WITH 'TESTSG' RETURN count(r)") == 1
    assert count(driver, "MATCH (:Company)-[r:SANCTIONED_AS]->(s {id: 'TEST-1'}) "
                         "RETURN count(r)") == 1

    records, _, _ = driver.execute_query(
        "MATCH (child:Company)-[:OWNED_BY]->(:Company)-[:SANCTIONED_AS]->(s) "
        "WHERE s.id STARTS WITH 'TEST-' RETURN child.lei AS lei"
    )
    assert [r["lei"] for r in records] == [CHILD]


def test_supply_links_and_news_load_and_prune(driver, tmp_path) -> None:
    import json

    from conftest import filing_row, mention_row, news_event, write_jsonl

    records = {PARENT: make_record(PARENT, "Test Customer Inc.", "US"),
               CHILD: make_record(CHILD, "Test Foundry Co., Ltd.", "TW"),
               CLEAN: make_record(CLEAN, "Test Lead Corp", "US")}
    write_raw_data(tmp_path, records, set(), [{"id": "TEST-1", "schema": "Company",
                                               "name": "Nobody", "last_seen": now_iso()}])
    raw = tmp_path / "raw"
    (raw / "gleif" / "seeds.json").write_text(json.dumps([PARENT]))
    write_jsonl(raw / "sec" / "filings.jsonl", [filing_row(PARENT, "999", "Test Customer")])
    write_jsonl(raw / "sec" / "supplier_mentions.jsonl", [
        mention_row(PARENT, "999", "Test Foundry Co., Ltd.", CHILD, "LLM"),
        mention_row(PARENT, "999", "Test Lead Corp", CLEAN, "RULES"),
    ])
    write_jsonl(raw / "gdelt" / "events.jsonl", [
        news_event("TEST-N1", ["test foundry"], ["FIRE"])])

    def build_and_load() -> None:
        c = warehouse.connect(":memory:")
        warehouse.build_all(c, raw / "gleif", raw / "opensanctions" / "targets.simple.csv",
                            raw / "sec", raw / "gdelt")
        graph.load_graph(c, driver)
        c.close()

    for _ in range(2):
        build_and_load()
    assert count(driver, f"MATCH ({{lei: '{CHILD}'}})-[r:SUPPLIES]->() RETURN count(r)") == 1
    assert count(driver, f"MATCH ({{lei: '{CLEAN}'}})-[r:POSSIBLE_SUPPLIER]->() "
                         "RETURN count(r)") == 1
    assert count(driver, f"MATCH ({{lei: '{CHILD}'}})-[:MENTIONED_IN]->(n:NewsEvent "
                         "{id: 'TEST-N1'}) RETURN count(n)") == 1
    records_, _, _ = driver.execute_query(
        f"MATCH (c:Company {{lei: '{CHILD}'}}) RETURN c.watchlist AS w")
    assert records_[0]["w"] is True

    # The extractor no longer finds the lead and the news window moved on: both disappear.
    write_jsonl(raw / "sec" / "supplier_mentions.jsonl", [
        mention_row(PARENT, "999", "Test Foundry Co., Ltd.", CHILD, "LLM")])
    write_jsonl(raw / "gdelt" / "events.jsonl", [])
    build_and_load()
    assert count(driver, "MATCH ()-[r:POSSIBLE_SUPPLIER]->(c) WHERE c.lei STARTS WITH "
                         "'TESTSG' RETURN count(r)") == 0
    assert count(driver, "MATCH (n:NewsEvent) WHERE n.id STARTS WITH 'TEST-' "
                         "RETURN count(n)") == 0
    assert count(driver, f"MATCH ({{lei: '{CHILD}'}})-[r:SUPPLIES]->() RETURN count(r)") == 1


def test_uniqueness_constraint_is_enforced(driver, con) -> None:
    from neo4j.exceptions import ConstraintError

    graph.load_graph(con, driver)
    with pytest.raises(ConstraintError):
        driver.execute_query("CREATE (:Company {lei: $lei})", lei=CLEAN)


def test_approved_analyst_actions_load_and_reset(driver, tmp_path, con) -> None:
    from supplygraph.agent.actions import (
        FLAG_SUPPLIER,
        REQUEST_REVIEW,
        ActionStore,
        build_actions_table,
    )

    store = ActionStore(tmp_path / "events.jsonl")
    for action, lei in [(FLAG_SUPPLIER, CHILD), (REQUEST_REVIEW, CHILD), (FLAG_SUPPLIER, CLEAN)]:
        a, _ = store.propose(con, action=action, lei=lei, sanction_id=None,
                             reason="parent is sanctioned", evidence=[f"lei:{lei}"],
                             proposed_by="agent:test")
        if lei == CHILD:  # CLEAN's flag stays a proposal
            store.decide(con, a.action_id, True, decided_by="ana")
    build_actions_table(con, store.path)
    graph.load_graph(con, driver)

    def props(lei: str):
        records, _, _ = driver.execute_query(
            "MATCH (c:Company {lei: $lei}) RETURN c.analyst_flags AS f, "
            "c.review_requested AS r", lei=lei)
        return records[0]["f"], records[0]["r"]

    assert props(CHILD) == ("parent is sanctioned", True)
    assert props(CLEAN) == (None, False)

    # Rebuilt without the approvals (e.g. a fresh log): the properties are cleared.
    build_actions_table(con, tmp_path / "empty.jsonl")
    graph.load_graph(con, driver)
    assert props(CHILD) == (None, False)


def test_cypher_tools_agree_with_the_warehouse_gold_and_cannot_write(driver, tmp_path) -> None:
    """The agent's multi-hop Cypher tools return exactly the gold the benchmark computes from
    the warehouse, and their sessions are read-only on the server side."""
    import json

    from neo4j import READ_ACCESS
    from neo4j.exceptions import ClientError

    from conftest import filing_row, mention_row, write_jsonl
    from supplygraph import graph_bench
    from supplygraph.agent.cypher import CypherTools

    sanc, sub, cust, sup, t2 = (make_lei(f"TESTSG00000000001{i}") for i in range(5))
    records = {sanc: make_record(sanc, "Test Sanctioned Refiner", "RU"),
               sub: make_record(sub, "Test Refiner Subsidiary", "DE"),
               cust: make_record(cust, "Test Customer Inc.", "US"),
               sup: make_record(sup, "Test Module Maker", "TW"),
               t2: make_record(t2, "Test Wafer Works", "TW")}
    write_raw_data(tmp_path, records, {(sub, sanc, gleif.DIRECT)},
                   [{"id": "TEST-9", "schema": "Company", "name": "Test Sanctioned Refiner",
                     "identifiers": sanc, "last_seen": now_iso()}])
    raw = tmp_path / "raw"
    (raw / "gleif" / "seeds.json").write_text(json.dumps([cust, sup]))
    write_jsonl(raw / "sec" / "filings.jsonl", [
        filing_row(cust, "901", "Test Customer", "0000000000-26-000901"),
        filing_row(sup, "902", "Test Module Maker", "0000000000-26-000902")])
    write_jsonl(raw / "sec" / "supplier_mentions.jsonl", [
        {**mention_row(cust, "901", "Test Sanctioned Refiner", sanc, "LLM"),
         "accession": "0000000000-26-000901"},
        {**mention_row(cust, "901", "Test Module Maker", sup, "RULES"),
         "accession": "0000000000-26-000901"},
        {**mention_row(sup, "902", "Test Wafer Works", t2, "RULES"),
         "accession": "0000000000-26-000902"}])
    con = warehouse.connect(":memory:")
    warehouse.build_all(con, raw / "gleif", raw / "opensanctions" / "targets.simple.csv",
                        raw / "sec")
    graph.load_graph(con, driver)

    tools, g = CypherTools(driver), graph_bench.WarehouseGraph(con)
    for lei in records:
        for family, spec in graph_bench.FAMILIES.items():
            gold, _ = graph_bench.gold_answer(g, family, lei)
            args = {spec["arg"]: spec["depth"]}
            out = tools.handlers[spec["tool"]](lei, **args)
            got = sorted(e["sanction_ref"] for e in out["exposures"]) \
                if family == "sanctions_exposure" else \
                sorted(i["ref"] for i in out.get("suppliers", out.get("affected", [])))
            assert got == gold, (family, records[lei]["attributes"]["entity"]["legalName"])
    # The interesting cases, spelled out: the customer of the sanctioned refiner is exposed,
    # its other supplier is not (risk does not flow customer -> supplier), and the wafer
    # maker is a tier-2 supplier reached only through a lead.
    assert tools.sanctions_exposure(cust)["exposed"] is True
    assert tools.sanctions_exposure(sup)["exposed"] is False
    tier2 = [s for s in tools.supplier_tree(cust)["suppliers"] if s["ref"] == f"lei:{t2}"]
    assert tier2 and tier2[0]["tier"] == 2 and tier2[0]["certainty"] == "lead"

    with driver.session(default_access_mode=READ_ACCESS) as session, \
            pytest.raises(ClientError):
        session.execute_read(lambda tx: tx.run("CREATE (:Company {lei: 'TESTSGWRITE'})")
                             .consume())
    assert count(driver, "MATCH (c {lei: 'TESTSGWRITE'}) RETURN count(c)") == 0
    con.close()
