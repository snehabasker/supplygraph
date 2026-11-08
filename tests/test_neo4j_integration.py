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
