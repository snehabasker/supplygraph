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
    "CREATE CONSTRAINT news_event_id IF NOT EXISTS FOR (n:NewsEvent) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.legal_name)",
    "CREATE INDEX company_country IF NOT EXISTS FOR (c:Company) ON (c.legal_country)",
    "CREATE INDEX news_published IF NOT EXISTS FOR (n:NewsEvent) ON (n.published_at)",
]

# Relationship types cannot be query parameters in Cypher, so one statement per link type.
_UPSERT_SUPPLY = """
UNWIND $rows AS r
MATCH (s:Company {{lei: r.supplier_lei}})
MATCH (c:Company {{lei: r.customer_lei}})
MERGE (s)-[l:{link_type}]->(c)
SET l.extraction_method = r.extraction_method, l.evidence = r.evidence,
    l.supplies = r.supplies, l.mentions = r.mentions, l.form = r.form,
    l.accession = r.accession, l.filing_date = r.filing_date
"""
SUPPLY_LINK_TYPES = ("SUPPLIES", "POSSIBLE_SUPPLIER")

SET_WATCHLIST = """
MATCH (c:Company) WHERE c.lei IN $leis
SET c.watchlist = c.lei IN $watch
"""

SET_RISK = """
UNWIND $rows AS r
MATCH (c:Company {lei: r.lei})
SET c.risk_score = r.risk_score, c.own_risk = r.own_risk,
    c.inherited_risk = r.inherited_risk, c.risk_drivers = r.drivers
"""

UPSERT_NEWS = """
UNWIND $rows AS r
MERGE (n:NewsEvent {id: r.id})
SET n += r
"""

UPSERT_MENTIONED_IN = """
UNWIND $rows AS r
MATCH (c:Company {lei: r.lei})
MATCH (n:NewsEvent {id: r.event_id})
MERGE (c)-[m:MENTIONED_IN]->(n)
SET m.matched_name = r.matched_name
"""

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
PRUNE_SUPPLY = """
MATCH (s:Company)-[l:{link_type}]->(c:Company)
WHERE c.lei IN $leis AND NOT [s.lei, c.lei] IN $keep DELETE l
"""
PRUNE_NEWS = [
    """MATCH (c:Company)-[m:MENTIONED_IN]->(n:NewsEvent)
       WHERE c.lei IN $leis AND NOT [c.lei, n.id] IN $keep DELETE m""",
    # Events that no longer mention any company (e.g. aged out of the news window).
    "MATCH (n:NewsEvent) WHERE NOT (n)<-[:MENTIONED_IN]-() DELETE n",
]


def _iso(value: object) -> object:
    """Neo4j's driver takes str/int/float/bool/None/lists; timestamps go in as ISO strings."""
    return value.isoformat() if hasattr(value, "isoformat") else value


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

    counts = {
        "companies": len(companies),
        "ownership": len(ownership),
        "sanctioned_entities": len(sanctioned),
        "sanction_matches": len(matches),
        "possible_matches": len(possible),
    }
    counts |= _load_supply_chain(con, driver, leis, batch_size)
    counts |= _load_news(con, driver, leis, batch_size)
    if table_exists(con, "company_risk"):
        risk = fetch_dicts(con, "SELECT lei, risk_score, own_risk, inherited_risk, drivers "
                                "FROM company_risk")
        _write(driver, SET_RISK, risk, batch_size, "risk scores")
        counts["risk_scores"] = sum(1 for r in risk if r["risk_score"] > 0)
    return counts


def _load_supply_chain(con: duckdb.DuckDBPyConnection, driver: Driver, leis: list[str],
                       batch_size: int) -> dict[str, int]:
    if not table_exists(con, "supply_links"):
        return {}
    counts = {}
    for link_type in SUPPLY_LINK_TYPES:
        rows = fetch_dicts(
            con,
            "SELECT supplier_lei, customer_lei, extraction_method, evidence, supplies, "
            "mentions, form, accession, filing_date FROM supply_links WHERE link_type = ?",
            [link_type],
        )
        _write(driver, _UPSERT_SUPPLY.format(link_type=link_type), rows, batch_size,
               f"{link_type} links")
        driver.execute_query(PRUNE_SUPPLY.format(link_type=link_type), leis=leis,
                             keep=[[r["supplier_lei"], r["customer_lei"]] for r in rows])
        counts[link_type.lower()] = len(rows)
    watch = [lei for (lei,) in con.execute("SELECT lei FROM watchlist").fetchall()] \
        if table_exists(con, "watchlist") else []
    driver.execute_query(SET_WATCHLIST, leis=leis, watch=watch)
    counts["watchlist"] = len(watch)
    return counts


def _load_news(con: duckdb.DuckDBPyConnection, driver: Driver, leis: list[str],
               batch_size: int) -> dict[str, int]:
    if not table_exists(con, "company_news"):
        return {}
    # Only events that mention a watchlist company are worth a node.
    events = [
        {k: _iso(v) for k, v in row.items()}
        for row in fetch_dicts(con, """
            SELECT e.id, e.published_at, e.title, e.url, e.domain, e.event_types, e.tone,
                   e.locations, e.copies
            FROM news_events e SEMI JOIN company_news n ON n.event_id = e.id
        """)
    ]
    links = fetch_dicts(con, "SELECT lei, event_id, matched_name FROM company_news")
    _write(driver, UPSERT_NEWS, events, batch_size, "news events")
    _write(driver, UPSERT_MENTIONED_IN, links, batch_size, "news mentions")
    driver.execute_query(PRUNE_NEWS[0], leis=leis,
                         keep=[[r["lei"], r["event_id"]] for r in links])
    driver.execute_query(PRUNE_NEWS[1])
    return {"news_events": len(events), "news_mentions": len(links)}
