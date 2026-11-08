"""Loads the warehouse into Neo4j following the ontology in docs/ontology.md.

All writes use MERGE, so loading twice produces the same graph (idempotent).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import duckdb
from neo4j import Driver

from supplygraph.warehouse import ORG_SCHEMAS, table_exists

log = logging.getLogger(__name__)

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT company_lei IF NOT EXISTS FOR (c:Company) REQUIRE c.lei IS UNIQUE",
    "CREATE CONSTRAINT sanctioned_id IF NOT EXISTS "
    "FOR (s:SanctionedEntity) REQUIRE s.id IS UNIQUE",
    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.legal_name)",
    "CREATE INDEX company_country IF NOT EXISTS FOR (c:Company) ON (c.legal_country)",
]

UPSERT_COMPANIES = """
UNWIND $rows AS r
MERGE (c:Company {lei: r.lei})
SET c += r
"""

UPSERT_OWNERSHIP = """
UNWIND $rows AS r
MATCH (child:Company {lei: r.child_lei})
MATCH (parent:Company {lei: r.parent_lei})
MERGE (child)-[o:OWNED_BY {relationship_type: r.relationship_type}]->(parent)
"""

UPSERT_SANCTIONED = """
UNWIND $rows AS r
MERGE (s:SanctionedEntity {id: r.id})
SET s += r
"""

UPSERT_MATCHES = """
UNWIND $rows AS r
MATCH (c:Company {lei: r.lei})
MATCH (s:SanctionedEntity {id: r.sanction_id})
MERGE (c)-[m:SANCTIONED_AS]->(s)
SET m.match_method = r.match_method, m.confidence = r.confidence, m.evidence = r.evidence
"""

UPSERT_POSSIBLE = """
UNWIND $rows AS r
MATCH (c:Company {lei: r.lei})
MATCH (s:SanctionedEntity {id: r.sanction_id})
MERGE (c)-[p:POSSIBLE_SANCTIONS_MATCH]->(s)
SET p.score = r.score, p.company_name = r.company_name, p.sanction_name = r.sanction_name,
    p.country_relation = r.country_relation
"""

# The graph mirrors the warehouse: links that ER no longer supports must disappear. Pruning is
# scoped to the companies in this load, so loading one dataset never deletes another's links.
PRUNE_STALE = [
    """MATCH (c:Company)-[m:SANCTIONED_AS]->(s:SanctionedEntity)
       WHERE c.lei IN $leis AND NOT [c.lei, s.id] IN $keep DELETE m""",
    """MATCH (c:Company)-[p:POSSIBLE_SANCTIONS_MATCH]->(s:SanctionedEntity)
       WHERE c.lei IN $leis AND NOT [c.lei, s.id] IN $keep DELETE p""",
]


def fetch_dicts(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> list[dict]:
    cur = con.execute(sql, params or [])
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def batched(rows: list[dict], size: int) -> Iterator[list[dict]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def _write(driver: Driver, query: str, rows: list[dict], batch_size: int, label: str) -> None:
    for batch in batched(rows, batch_size):
        driver.execute_query(query, rows=batch)
    log.info("Neo4j: upserted %d %s", len(rows), label)


def load_graph(con: duckdb.DuckDBPyConnection, driver: Driver, batch_size: int = 1000) -> dict:
    for stmt in SCHEMA_STATEMENTS:
        driver.execute_query(stmt)

    companies = fetch_dicts(con, "SELECT * EXCLUDE (source) FROM companies")
    ownership = fetch_dicts(con, "SELECT * FROM ownership")
    placeholders = ", ".join("?" * len(ORG_SCHEMAS))
    sanctioned = fetch_dicts(
        con,
        f"SELECT id, schema, name, countries, sanctions, program_ids, dataset, last_seen "
        f"FROM sanctions WHERE schema IN ({placeholders}) "
        f"OR id IN (SELECT sanction_id FROM company_sanction_matches)",
        list(ORG_SCHEMAS),
    )
    matches = fetch_dicts(con, "SELECT * FROM company_sanction_matches")
    possible = []
    if table_exists(con, "er_candidates"):
        possible = fetch_dicts(
            con,
            "SELECT lei, sanction_id, score, company_name, sanction_name, country_relation "
            "FROM er_candidates WHERE decision = 'review'",
        )

    _write(driver, UPSERT_COMPANIES, companies, batch_size, "companies")
    _write(driver, UPSERT_OWNERSHIP, ownership, batch_size, "ownership edges")
    _write(driver, UPSERT_SANCTIONED, sanctioned, batch_size, "sanctioned entities")
    _write(driver, UPSERT_MATCHES, matches, batch_size, "sanction matches")
    _write(driver, UPSERT_POSSIBLE, possible, batch_size, "possible matches (review queue)")
    leis = [c["lei"] for c in companies]
    driver.execute_query(PRUNE_STALE[0], leis=leis,
                         keep=[[m["lei"], m["sanction_id"]] for m in matches])
    driver.execute_query(PRUNE_STALE[1], leis=leis,
                         keep=[[p["lei"], p["sanction_id"]] for p in possible])
    return {
        "companies": len(companies),
        "ownership": len(ownership),
        "sanctioned_entities": len(sanctioned),
        "sanction_matches": len(matches),
        "possible_matches": len(possible),
    }
