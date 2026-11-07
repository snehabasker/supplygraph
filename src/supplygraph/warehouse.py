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
    "lei", "legal_name", "jurisdiction", "legal_country", "legal_city", "hq_country",
    "hq_city", "category", "legal_form_id", "entity_status", "registration_status",
    "last_update",
]

# Sanctions entity types that can plausibly be the same real-world thing as a company.
ORG_SCHEMAS = ("Company", "Organization", "LegalEntity")


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


def build_gleif(con: duckdb.DuckDBPyConnection, gleif_dir: Path) -> tuple[int, int]:
    records = _read_jsonl(gleif_dir / "records.jsonl")
    rels = _read_jsonl(gleif_dir / "relationships.jsonl")

    cols_sql = ", ".join(f"{c} VARCHAR" for c in COMPANY_COLUMNS)
    con.execute(f"CREATE OR REPLACE TABLE companies ({cols_sql}, source VARCHAR)")
    rows = [[parse_record(r)[c] for c in COMPANY_COLUMNS] + ["GLEIF"] for r in records]
    if rows:
        placeholders = ", ".join("?" * (len(COMPANY_COLUMNS) + 1))
        con.executemany(f"INSERT INTO companies VALUES ({placeholders})", rows)

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


def build_matches(con: duckdb.DuckDBPyConnection) -> int:
    """Deterministic, zero-false-positive linking: a sanctions record that lists a
    company's LEI among its identifiers. (Fuzzy entity resolution comes in Milestone 3.)"""
    con.execute("""
        CREATE OR REPLACE TABLE company_sanction_matches AS
        WITH ids AS (
            SELECT id AS sanction_id, trim(unnest(string_split(identifiers, ';'))) AS identifier
            FROM sanctions
            WHERE identifiers IS NOT NULL AND identifiers <> ''
        )
        -- DOUBLE, not the DECIMAL a bare 1.0 literal yields: the Neo4j driver rejects Decimal.
        SELECT DISTINCT c.lei, ids.sanction_id, 'LEI_IDENTIFIER' AS match_method,
               CAST(1.0 AS DOUBLE) AS confidence
        FROM companies c
        JOIN ids ON ids.identifier = c.lei
    """)
    row = con.execute("SELECT count(*) FROM company_sanction_matches").fetchone()
    count = row[0] if row else 0
    log.info("company_sanction_matches=%d", count)
    return count


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
