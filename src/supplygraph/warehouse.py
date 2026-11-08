"""DuckDB warehouse: turns raw (bronze) files into clean, typed tables (silver).

Every build is a full refresh (CREATE OR REPLACE), so re-running is idempotent.
"""

from __future__ import annotations

import json
import logging
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
    records = _read_jsonl(gleif_dir / "records.jsonl")
    rels = _read_jsonl(gleif_dir / "relationships.jsonl")

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


def build_all(con: duckdb.DuckDBPyConnection, gleif_dir: Path, sanctions_csv: Path) -> dict:
    companies, ownership = build_gleif(con, gleif_dir)
    sanctions = build_sanctions(con, sanctions_csv)
    matches = build_matches(con)
    return {
        "companies": companies,
        "ownership": ownership,
        "sanctions": sanctions,
        "company_sanction_matches": matches,
    }
