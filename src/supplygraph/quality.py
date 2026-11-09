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
    Check("matches_method_known", "company_sanction_matches", ERROR,
          "Every match has a known match_method",
          sql="SELECT count(*) FROM company_sanction_matches WHERE match_method NOT IN "
              "('LEI_IDENTIFIER', 'REGISTRATION_NUMBER', 'FUZZY_NAME', 'LLM_ADJUDICATED', "
              "'ANALYST_CONFIRMED')"),
    Check("matches_confidence_rule", "company_sanction_matches", ERROR,
          "Identifier and analyst-confirmed matches have confidence 1.0; automatic name-based "
          "matches < 1.0",
          sql="""SELECT count(*) FROM company_sanction_matches WHERE
                 (match_method IN ('LEI_IDENTIFIER', 'REGISTRATION_NUMBER', 'ANALYST_CONFIRMED')
                  AND confidence <> 1.0)
                 OR (match_method IN ('FUZZY_NAME', 'LLM_ADJUDICATED')
                     AND NOT (confidence > 0 AND confidence < 1.0))"""),
    Check("matches_have_evidence", "company_sanction_matches", ERROR,
          "Every match records its evidence (auditability)",
          sql="SELECT count(*) FROM company_sanction_matches WHERE evidence IS NULL "
              "OR NOT json_valid(evidence)"),
    Check("matches_reference_known_entities", "company_sanction_matches", ERROR,
          "Matches point at existing companies and sanctions records",
          sql="""SELECT count(*) FROM company_sanction_matches m
                 LEFT JOIN companies c ON m.lei = c.lei
                 LEFT JOIN sanctions s ON m.sanction_id = s.id
                 WHERE c.lei IS NULL OR s.id IS NULL"""),
    Check("er_decisions_valid", "er_candidates", ERROR,
          "Entity-resolution decisions are match / no_match / review",
          sql="SELECT count(*) FROM er_candidates "
              "WHERE decision NOT IN ('match', 'no_match', 'review')"),
    Check("filings_accession_unique", "filings", ERROR, "One manifest row per filing",
          sql="SELECT count(*) - count(DISTINCT accession) FROM filings"),
    Check("mentions_method_known", "supplier_mentions", ERROR,
          "Every supplier mention comes from a known extraction method",
          sql="SELECT count(*) FROM supplier_mentions WHERE method NOT IN ('RULES', 'LLM')"),
    Check("mentions_have_evidence", "supplier_mentions", ERROR,
          "Every supplier mention quotes its evidence sentence",
          sql="SELECT count(*) FROM supplier_mentions "
              "WHERE evidence IS NULL OR trim(evidence) = ''"),
    Check("mentions_resolution_rate", "supplier_mentions", WARN,
          "At least half of the supplier names resolve to an LEI",
          sql="SELECT CASE WHEN count(*) > 0 AND count(supplier_lei) * 2 < count(*) "
              "THEN 1 ELSE 0 END FROM supplier_mentions"),
    Check("supply_links_no_orphans", "supply_links", ERROR,
          "Both ends of every supplier link are known companies",
          sql="""SELECT count(*) FROM supply_links l
                 LEFT JOIN companies s ON l.supplier_lei = s.lei
                 LEFT JOIN companies c ON l.customer_lei = c.lei
                 WHERE s.lei IS NULL OR c.lei IS NULL"""),
    Check("supply_links_no_self_loops", "supply_links", ERROR, "No company supplies itself",
          sql="SELECT count(*) FROM supply_links WHERE supplier_lei = customer_lei"),
    Check("supply_links_type_follows_method", "supply_links", ERROR,
          "LLM links are SUPPLIES, rule links are POSSIBLE_SUPPLIER (ontology v0.3)",
          sql="""SELECT count(*) FROM supply_links WHERE link_type IS NULL
                 OR (extraction_method = 'LLM' AND link_type <> 'SUPPLIES')
                 OR (extraction_method = 'RULES' AND link_type <> 'POSSIBLE_SUPPLIER')"""),
    Check("supply_links_have_evidence", "supply_links", ERROR,
          "Every supplier link quotes its evidence sentence",
          sql="SELECT count(*) FROM supply_links WHERE evidence IS NULL OR trim(evidence) = ''"),
    Check("news_events_typed", "news_events", ERROR,
          "Every news event has at least one disruption event type",
          sql="SELECT count(*) FROM news_events "
              "WHERE event_types IS NULL OR len(event_types) = 0"),
    Check("news_id_unique", "news_events", ERROR, "GDELT record ids are unique",
          sql="SELECT count(*) - count(DISTINCT id) FROM news_events"),
    Check("risk_scores_bounded", "company_risk", ERROR,
          "Risk scores are in [0, 1] and own + inherited = total",
          sql="""SELECT count(*) FROM company_risk WHERE risk_score < 0 OR risk_score > 1
                 OR own_risk < 0 OR own_risk > risk_score + 1e-9
                 OR abs(own_risk + inherited_risk - risk_score) > 1e-5"""),
    Check("risk_scores_explained", "company_risk", ERROR,
          "Every non-zero risk score lists the signals and paths that produced it",
          sql="SELECT count(*) FROM company_risk WHERE risk_score > 0 "
              "AND (drivers IS NULL OR NOT json_valid(drivers) OR json_array_length(drivers) = 0)"),
    Check("risk_covers_companies", "company_risk", ERROR, "Every company has a risk row",
          sql="SELECT count(*) FROM companies c ANTI JOIN company_risk r USING (lei)"),
    Check("actions_valid", "analyst_actions", ERROR,
          "Analyst actions have a known type and status, a reason and evidence",
          sql="""SELECT count(*) FROM analyst_actions WHERE
                 action NOT IN ('flag_supplier', 'request_review', 'confirm_match',
                                'dismiss_match')
                 OR status NOT IN ('proposed', 'approved', 'rejected')
                 OR reason IS NULL OR trim(reason) = '' OR proposed_by IS NULL
                 OR evidence IS NULL OR NOT json_valid(evidence)
                 OR json_array_length(evidence) = 0"""),
    Check("actions_decisions_audited", "analyst_actions", ERROR,
          "Every approved or rejected action records who decided and when",
          sql="""SELECT count(*) FROM analyst_actions WHERE status <> 'proposed'
                 AND (decided_by IS NULL OR trim(decided_by) = '' OR decided_at IS NULL)"""),
    Check("actions_reference_known_companies", "analyst_actions", WARN,
          "Open and approved actions point at companies still in the graph",
          sql="""SELECT count(*) FROM analyst_actions a ANTI JOIN companies c USING (lei)
                 WHERE a.status <> 'rejected'"""),
    Check("company_news_references_valid", "company_news", ERROR,
          "News links point from watchlist companies to existing events",
          sql="""SELECT count(*) FROM company_news n
                 LEFT JOIN watchlist w ON n.lei = w.lei
                 LEFT JOIN news_events e ON n.event_id = e.id
                 WHERE w.lei IS NULL OR e.id IS NULL"""),
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
