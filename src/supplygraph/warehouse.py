"""DuckDB warehouse: turns raw (bronze) files into clean, typed tables (silver).

Every build is a full refresh (CREATE OR REPLACE), so re-running is idempotent.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import duckdb

from supplygraph.ingest.gleif import parse_record

log = logging.getLogger(__name__)

COMPANY_COLUMNS = [
    "lei", "legal_name", "other_names", "registered_as", "jurisdiction", "legal_country",
    "legal_city", "hq_country", "hq_city", "category", "legal_form_id", "entity_status",
    "registration_status", "last_update",
]

# Sanctions entity types that can plausibly be the same real-world thing as a company.
ORG_SCHEMAS = ("Company", "Organization", "LegalEntity")

# Identifier-based match methods: deterministic, confidence 1.0.
IDENTIFIER_METHODS = ("LEI_IDENTIFIER", "REGISTRATION_NUMBER")
MIN_REGISTRATION_NUMBER_LENGTH = 6


def connect(path: Path | str) -> duckdb.DuckDBPyConnection:
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path = str(path)
    return duckdb.connect(path)


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and row[0])


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def company_rows(records: list[dict]) -> list[list]:
    return [[parse_record(r)[c] for c in COMPANY_COLUMNS] + ["GLEIF"] for r in records]


def build_gleif(con: duckdb.DuckDBPyConnection, gleif_dir: Path) -> tuple[int, int]:
    """Seed crawl plus, if present, the suppliers found in filings (`gleif_dir/suppliers`)."""
    records = _read_jsonl(gleif_dir / "records.jsonl")
    rels = _read_jsonl(gleif_dir / "relationships.jsonl")
    supplier_dir = gleif_dir / "suppliers"
    if (supplier_dir / "records.jsonl").exists():
        records += _read_jsonl(supplier_dir / "records.jsonl")
        rels += _read_jsonl(supplier_dir / "relationships.jsonl")
    records = list({r["id"]: r for r in reversed(records)}.values())  # first occurrence wins
    rels = [dict(t) for t in dict.fromkeys(tuple(sorted(r.items())) for r in rels)]

    rows = company_rows(records)
    create_companies_table(con, "companies", rows)

    con.execute(
        "CREATE OR REPLACE TABLE ownership "
        "(child_lei VARCHAR, parent_lei VARCHAR, relationship_type VARCHAR)"
    )
    if rels:
        con.executemany(
            "INSERT INTO ownership VALUES (?, ?, ?)",
            [[r["child_lei"], r["parent_lei"], r["relationship_type"]] for r in rels],
        )
    log.info("companies=%d ownership=%d", len(rows), len(rels))
    return len(rows), len(rels)


def build_sanctions(con: duckdb.DuckDBPyConnection, csv_path: Path) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found; run `supplygraph ingest-sanctions` first")
    path_literal = csv_path.resolve().as_posix().replace("'", "''")
    con.execute(f"""
        CREATE OR REPLACE TABLE sanctions AS
        SELECT id, schema, name, aliases, countries, identifiers, sanctions,
               program_ids, dataset, first_seen, last_seen
        FROM read_csv('{path_literal}', header = true, all_varchar = true)
    """)
    row = con.execute("SELECT count(*) FROM sanctions").fetchone()
    count = row[0] if row else 0
    log.info("sanctions=%d", count)
    return count


def build_matches(
    con: duckdb.DuckDBPyConnection,
    companies_table: str = "companies",
    out_table: str = "company_sanction_matches",
    temp: bool = False,
) -> int:
    """Deterministic identifier linking: a sanctions record that lists the company's LEI, or
    its local registration number in the same country. Fuzzy name matching lives in `er/`.

    `companies_table` / `out_table` let the ER evaluation label its gold set with exactly the
    same logic (table names are internal constants, never user input)."""
    kind = "TEMP TABLE" if temp else "TABLE"
    con.execute(f"""
        CREATE OR REPLACE {kind} {out_table} AS
        WITH ids AS (
            SELECT id AS sanction_id,
                   string_split(lower(coalesce(countries, '')), ';') AS countries,
                   trim(unnest(string_split(identifiers, ';'))) AS identifier
            FROM sanctions
            WHERE identifiers IS NOT NULL AND identifiers <> ''
        ),
        by_lei AS (
            SELECT c.lei, ids.sanction_id, 'LEI_IDENTIFIER' AS match_method,
                   json_object('lei', c.lei)::VARCHAR AS evidence
            FROM {companies_table} c JOIN ids ON ids.identifier = c.lei
        ),
        -- Registration numbers are only unique within a country, so the countries must agree,
        -- and very short numbers are ignored to avoid accidental collisions.
        by_registration AS (
            SELECT c.lei, ids.sanction_id, 'REGISTRATION_NUMBER' AS match_method,
                   json_object('registered_as', c.registered_as,
                               'country', c.legal_country)::VARCHAR AS evidence
            FROM {companies_table} c
            JOIN ids ON ids.identifier = c.registered_as
            WHERE length(c.registered_as) >= {MIN_REGISTRATION_NUMBER_LENGTH}
              AND list_contains(ids.countries, lower(c.legal_country))
        )
        -- DOUBLE, not the DECIMAL a bare 1.0 literal yields: the Neo4j driver rejects Decimal.
        SELECT DISTINCT lei, sanction_id, match_method, CAST(1.0 AS DOUBLE) AS confidence, evidence
        FROM by_lei
        UNION ALL
        SELECT DISTINCT r.lei, r.sanction_id, r.match_method, CAST(1.0 AS DOUBLE), r.evidence
        FROM by_registration r ANTI JOIN by_lei l USING (lei, sanction_id)
    """)
    row = con.execute(f"SELECT count(*) FROM {out_table}").fetchone()
    count = row[0] if row else 0
    log.info("%s=%d", out_table, count)
    return count


def create_companies_table(
    con: duckdb.DuckDBPyConnection, name: str, rows: list[list], temp: bool = False
) -> None:
    kind = "TEMP TABLE" if temp else "TABLE"
    cols_sql = ", ".join(f"{c} VARCHAR" for c in COMPANY_COLUMNS)
    con.execute(f"CREATE OR REPLACE {kind} {name} ({cols_sql}, source VARCHAR)")
    if rows:
        placeholders = ", ".join("?" * (len(COMPANY_COLUMNS) + 1))
        con.executemany(f"INSERT INTO {name} VALUES ({placeholders})", rows)


# Extraction method -> graph link type (see docs/ontology.md: link type follows precision).
SUPPLIER_LINK_TYPES = {"LLM": "SUPPLIES", "RULES": "POSSIBLE_SUPPLIER"}

FILING_COLUMNS = ["filer_lei", "cik", "filer_name", "entity_name", "form", "accession",
                  "filing_date", "report_date", "url", "path"]
MENTION_COLUMNS = ["filer_cik", "filer_lei", "passage_id", "supplier_name", "name_as_written",
                   "supplies", "evidence", "method", "model", "accession", "form",
                   "filing_date", "supplier_lei", "resolution_status", "resolved_legal_name"]


def _create_table(con: duckdb.DuckDBPyConnection, name: str, columns: list[str],
                  rows: list[dict]) -> int:
    con.execute(f"CREATE OR REPLACE TABLE {name} ({', '.join(f'{c} VARCHAR' for c in columns)})")
    if rows:
        con.executemany(f"INSERT INTO {name} VALUES ({', '.join('?' * len(columns))})",
                        [[r.get(c) for c in columns] for r in rows])
    return len(rows)


def build_supply_chain(con: duckdb.DuckDBPyConnection, sec_dir: Path) -> dict[str, int]:
    """Filings, supplier mentions and the supplier links derived from them. Missing inputs
    (SEC ingestion not run) produce empty tables, so the rest of the pipeline still works."""
    filings_path, mentions_path = sec_dir / "filings.jsonl", sec_dir / "supplier_mentions.jsonl"
    filings = _create_table(con, "filings", FILING_COLUMNS,
                            _read_jsonl(filings_path) if filings_path.exists() else [])
    mentions = _create_table(con, "supplier_mentions", MENTION_COLUMNS,
                             _read_jsonl(mentions_path) if mentions_path.exists() else [])
    cases = " ".join(f"WHEN '{m}' THEN '{t}'" for m, t in SUPPLIER_LINK_TYPES.items())
    con.execute(f"""
        CREATE OR REPLACE TABLE supply_links AS
        SELECT m.supplier_lei, m.filer_lei AS customer_lei, m.method AS extraction_method,
               CASE m.method {cases} END AS link_type,
               CAST(count(*) AS BIGINT) AS mentions,
               string_agg(DISTINCT m.supplies, '; ') AS supplies,
               arg_min(m.evidence, length(m.evidence)) AS evidence,
               any_value(m.supplier_name) AS supplier_name,
               any_value(m.form) AS form, any_value(m.accession) AS accession,
               any_value(m.filing_date) AS filing_date
        FROM supplier_mentions m
        JOIN companies s ON s.lei = m.supplier_lei
        JOIN companies c ON c.lei = m.filer_lei
        WHERE m.supplier_lei IS NOT NULL AND m.supplier_lei <> m.filer_lei
        GROUP BY m.supplier_lei, m.filer_lei, m.method
    """)
    row = con.execute("SELECT count(*) FROM supply_links").fetchone()
    links = row[0] if row else 0
    log.info("filings=%d supplier_mentions=%d supply_links=%d", filings, mentions, links)
    return {"filings": filings, "supplier_mentions": mentions, "supply_links": links}


def build_watchlist(con: duckdb.DuckDBPyConnection, gleif_dir: Path) -> int:
    """Companies we monitor: seeds, SEC filers, both ends of every *asserted* supplier link,
    and companies an analyst flagged (approved `flag_supplier` actions).
    POSSIBLE_SUPPLIER leads don't expand the watchlist: on real filings they put Microsoft
    and OpenAI (a partner and a customer) under supplier-disruption monitoring."""
    seeds_path = gleif_dir / "seeds.json"
    seeds = json.loads(seeds_path.read_text(encoding="utf-8")) if seeds_path.exists() else []
    con.execute("CREATE OR REPLACE TEMP TABLE _seeds (lei VARCHAR)")
    if seeds:
        con.executemany("INSERT INTO _seeds VALUES (?)", [[s] for s in seeds])
    flags = ("UNION ALL SELECT lei, 'analyst_flag' FROM analyst_actions "
             "WHERE action = 'flag_supplier' AND status = 'approved'"
             if table_exists(con, "analyst_actions") else "")
    con.execute(f"""
        CREATE OR REPLACE TABLE watchlist AS
        SELECT lei, string_agg(DISTINCT reason, ';' ORDER BY reason) AS reasons FROM (
            SELECT lei, 'seed' AS reason FROM _seeds
            UNION ALL SELECT filer_lei, 'sec_filer' FROM filings
            UNION ALL SELECT supplier_lei, 'supplier' FROM supply_links
                      WHERE link_type = 'SUPPLIES'
            UNION ALL SELECT customer_lei, 'customer' FROM supply_links
                      WHERE link_type = 'SUPPLIES'
            {flags}
        ) w SEMI JOIN companies c USING (lei)
        GROUP BY lei
    """)
    row = con.execute("SELECT count(*) FROM watchlist").fetchone()
    return row[0] if row else 0


NEWS_COLUMNS = {
    "id": "VARCHAR", "published_at": "TIMESTAMPTZ", "title": "VARCHAR", "url": "VARCHAR",
    "domain": "VARCHAR", "event_types": "VARCHAR[]", "organizations": "VARCHAR[]",
    "tone": "DOUBLE", "locations": "VARCHAR[]", "copies": "INTEGER",
}


def dedupe_syndicated(events: list[dict]) -> list[dict]:
    """One story is often republished by dozens of outlets, each a separate GKG record. Keep
    the earliest copy per title (organizations and event types are unioned) and count copies."""
    groups: dict[str, list[dict]] = {}
    for e in sorted(events, key=lambda e: (e.get("published_at") or "", e["id"])):
        title = " ".join((e.get("title") or "").lower().split())
        groups.setdefault(title or e["id"], []).append(e)
    out = []
    for copies in groups.values():
        first = dict(copies[0])
        first["organizations"] = list(dict.fromkeys(o for c in copies for o in c["organizations"]))
        first["event_types"] = sorted({t for c in copies for t in c["event_types"]})
        first["copies"] = len(copies)
        out.append(first)
    return out


def build_news(con: duckdb.DuckDBPyConnection, gdelt_dir: Path, sec_dir: Path) -> dict[str, int]:
    from supplygraph.news import match_events
    from supplygraph.news_cluster import merge_stories

    events_path = gdelt_dir / "events.jsonl"
    raw_events = _read_jsonl(events_path) if events_path.exists() else []
    deduped = dedupe_syndicated(raw_events)
    events = merge_stories(deduped)  # near-duplicate headlines of one story
    columns = ", ".join(f"{name} {kind}" for name, kind in NEWS_COLUMNS.items())
    con.execute(f"CREATE OR REPLACE TABLE news_events ({columns})")
    if events:
        # Bulk load through DuckDB's JSON reader: executemany inserts row by row and took
        # minutes for a day of news.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for e in events:
                    fh.write(json.dumps({k: e.get(k) for k in NEWS_COLUMNS},
                                        ensure_ascii=False) + "\n")
            spec = ", ".join(f"'{name}': '{kind}'" for name, kind in NEWS_COLUMNS.items())
            literal = path.as_posix().replace("'", "''")
            con.execute(f"INSERT INTO news_events SELECT * FROM read_json('{literal}', "
                        f"format = 'newline_delimited', columns = {{{spec}}})")
    aliases_path = sec_dir / "aliases.json"
    aliases = (json.loads(aliases_path.read_text(encoding="utf-8"))
               if aliases_path.exists() else {})
    links, stats = match_events(con, events, aliases)
    con.execute("CREATE OR REPLACE TABLE company_news "
                "(lei VARCHAR, event_id VARCHAR, matched_name VARCHAR)")
    if links:
        con.executemany("INSERT INTO company_news VALUES (?, ?, ?)", links)
    log.info("news_events=%d (%d raw, %d after exact-title dedupe) company_news=%d (%s)",
             len(events), len(raw_events), len(deduped), len(links), stats)
    return {"news_events": len(events), "raw_news_events": len(raw_events),
            "news_events_after_title_dedupe": len(deduped),
            "company_news": len(links), **stats}


def build_all(
    con: duckdb.DuckDBPyConnection,
    gleif_dir: Path,
    sanctions_csv: Path,
    sec_dir: Path | None = None,
    gdelt_dir: Path | None = None,
    actions_log: Path | None = None,
) -> dict:
    companies, ownership = build_gleif(con, gleif_dir)
    sanctions = build_sanctions(con, sanctions_csv)
    matches = build_matches(con)
    counts = {
        "companies": companies,
        "ownership": ownership,
        "sanctions": sanctions,
        "company_sanction_matches": matches,
    }
    if actions_log is not None:  # before the watchlist: approved flags extend it
        from supplygraph.agent.actions import build_actions_table

        counts["analyst_actions"] = build_actions_table(con, actions_log)
    if sec_dir is not None:
        counts |= build_supply_chain(con, sec_dir)
        counts["watchlist"] = build_watchlist(con, gleif_dir)
        if gdelt_dir is not None:
            news = build_news(con, gdelt_dir, sec_dir)
            counts |= {"news_events": news["news_events"], "company_news": news["company_news"]}
    return counts
