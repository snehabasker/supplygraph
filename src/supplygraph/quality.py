"""Data quality checks. Each check counts failing rows; `error` checks gate the pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import duckdb

from supplygraph.ingest.gleif import is_valid_lei
from supplygraph.warehouse import table_exists

ERROR = "error"
WARN = "warn"


@dataclass(frozen=True)
class Check:
    name: str
    table: str
    severity: str
    description: str
    sql: str | None = None  # must return a single integer: the number of failing rows
    func: Callable[[duckdb.DuckDBPyConnection], int] | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: str
    status: str  # "pass" | "fail" | "skipped"
    failures: int
    description: str


def _bad_lei_checksums(con: duckdb.DuckDBPyConnection) -> int:
    return sum(1 for (lei,) in con.execute("SELECT lei FROM companies").fetchall()
               if not is_valid_lei(lei))


CHECKS: list[Check] = [
    Check("companies_not_empty", "companies", ERROR, "At least one company was loaded",
          sql="SELECT CASE WHEN count(*) = 0 THEN 1 ELSE 0 END FROM companies"),
    Check("companies_lei_unique", "companies", ERROR, "LEI is a unique key",
          sql="SELECT count(*) - count(DISTINCT lei) FROM companies"),
    Check("companies_lei_valid", "companies", ERROR, "LEI passes ISO 17442 check digits",
          func=_bad_lei_checksums),
    Check("companies_name_present", "companies", ERROR, "Every company has a legal name",
          sql="SELECT count(*) FROM companies WHERE legal_name IS NULL OR trim(legal_name) = ''"),
    Check("companies_country_iso2", "companies", WARN, "Legal country is ISO 3166 alpha-2",
          sql="SELECT count(*) FROM companies WHERE legal_country IS NULL "
              "OR NOT regexp_full_match(legal_country, '[A-Z]{2}')"),
    Check("ownership_no_orphans", "ownership", ERROR, "Both ends of every edge are known companies",
          sql="""SELECT count(*) FROM ownership o
                 LEFT JOIN companies c ON o.child_lei = c.lei
                 LEFT JOIN companies p ON o.parent_lei = p.lei
                 WHERE c.lei IS NULL OR p.lei IS NULL"""),
    Check("ownership_no_self_loops", "ownership", ERROR, "No company owns itself",
          sql="SELECT count(*) FROM ownership WHERE child_lei = parent_lei"),
    Check("ownership_type_valid", "ownership", ERROR, "Relationship type is DIRECT or ULTIMATE",
          sql="SELECT count(*) FROM ownership "
              "WHERE relationship_type NOT IN ('DIRECT', 'ULTIMATE')"),
    Check("sanctions_not_empty", "sanctions", ERROR, "Sanctions list was loaded",
          sql="SELECT CASE WHEN count(*) = 0 THEN 1 ELSE 0 END FROM sanctions"),
    Check("sanctions_id_unique", "sanctions", ERROR, "Sanctions entity id is unique",
          sql="SELECT count(*) - count(DISTINCT id) FROM sanctions"),
    Check("sanctions_name_present", "sanctions", WARN, "Every sanctions entity has a name",
          sql="SELECT count(*) FROM sanctions WHERE name IS NULL OR trim(name) = ''"),
    Check("sanctions_fresh", "sanctions", WARN, "Sanctions data seen within the last 7 days",
          sql="SELECT CASE WHEN max(TRY_CAST(last_seen AS TIMESTAMP)) "
              ">= now()::TIMESTAMP - INTERVAL 7 DAY THEN 0 ELSE 1 END FROM sanctions"),
]


def run_checks(con: duckdb.DuckDBPyConnection, checks: list[Check] = CHECKS) -> list[CheckResult]:
    results = []
    for check in checks:
        if not table_exists(con, check.table):
            results.append(CheckResult(check.name, check.severity, "skipped", 0, check.description))
            continue
        if check.func is not None:
            failures = check.func(con)
        else:
            row = con.execute(check.sql).fetchone()
            failures = int(row[0]) if row and row[0] is not None else 0
        status = "pass" if failures == 0 else "fail"
        results.append(CheckResult(check.name, check.severity, status, failures, check.description))
    return results


def has_blocking_failures(results: list[CheckResult]) -> bool:
    return any(r.status == "fail" and r.severity == ERROR for r in results)
