"""Production entity resolution: companies x sanctions -> matches, review queue, audit trail.

Decision policy for each company's best candidate:

    score >= auto_match            -> MATCH  (FUZZY_NAME)
    review <= score < auto_match   -> LLM decides (LLM_ADJUDICATED), or NEEDS_REVIEW if no LLM
                                      or the LLM call fails
    score < review                 -> ignored
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import duckdb

from supplygraph.er.adjudicate import Adjudicator, Decision
from supplygraph.er.matcher import Candidate, Matcher, MatcherConfig, Party
from supplygraph.warehouse import IDENTIFIER_METHODS, ORG_SCHEMAS

log = logging.getLogger(__name__)

MATCH = "match"
NO_MATCH = "no_match"
REVIEW = "review"


@dataclass(frozen=True)
class Thresholds:
    # Chosen on the dev split by `supplygraph evaluate-er` (auto_match = lowest threshold
    # with >= 95% dev precision). Re-run the evaluation before changing these.
    auto_match: float = 0.97
    review: float = 0.90


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def _countries(*values: str | None) -> frozenset[str]:
    out = set()
    for value in values:
        for code in _split(value):
            if len(code) == 2 and code.isalpha():
                out.add(code.upper())
    return frozenset(out)


def company_party(lei: str, legal_name: str | None, other_names: str | None,
                  legal_country: str | None, hq_country: str | None) -> Party:
    names = tuple(dict.fromkeys(n for n in [legal_name, *_split(other_names)] if n))
    return Party(lei, names, _countries(legal_country, hq_country))


def load_company_parties(con: duckdb.DuckDBPyConnection) -> list[Party]:
    rows = con.execute(
        "SELECT lei, legal_name, other_names, legal_country, hq_country FROM companies"
    ).fetchall()
    return [company_party(*row) for row in rows]


def load_sanction_parties(con: duckdb.DuckDBPyConnection) -> list[Party]:
    placeholders = ", ".join("?" * len(ORG_SCHEMAS))
    rows = con.execute(
        f"SELECT id, name, aliases, countries FROM sanctions WHERE schema IN ({placeholders})",
        list(ORG_SCHEMAS),
    ).fetchall()
    return [
        Party(sid, tuple(dict.fromkeys(n for n in [name, *_split(aliases)] if n)),
              _countries(countries))
        for sid, name, aliases, countries in rows
    ]


def best_candidates(companies: list[Party], matcher: Matcher, min_score: float) -> list[Candidate]:
    """Top candidate per company."""
    found = []
    for company in companies:
        best = matcher.best(company, min_score)
        if best is not None:
            found.append(best)
    return found


LLM_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS er_llm_decisions (
    lei VARCHAR, sanction_id VARCHAR, model VARCHAR,
    is_match BOOLEAN, confidence DOUBLE, reasoning VARCHAR, decided_at TIMESTAMP,
    PRIMARY KEY (lei, sanction_id, model)
)
"""


class CachedAdjudicator:
    """Wraps an Adjudicator with a persistent DuckDB cache."""

    def __init__(self, inner: Adjudicator, con: duckdb.DuckDBPyConnection) -> None:
        self.inner = inner
        self.model = inner.model
        self.con = con
        con.execute(LLM_CACHE_DDL)
        self.calls = 0
        self.hits = 0

    def adjudicate(self, company: Party, sanction: Party, candidate: Candidate) -> Decision | None:
        row = self.con.execute(
            "SELECT is_match, confidence, reasoning FROM er_llm_decisions "
            "WHERE lei = ? AND sanction_id = ? AND model = ?",
            [company.id, sanction.id, self.model],
        ).fetchone()
        if row:
            self.hits += 1
            return Decision(bool(row[0]), float(row[1]), row[2], self.model)
        self.calls += 1
        decision = self.inner.adjudicate(company, sanction, candidate)
        if decision is not None:  # failures are not cached, so they are retried next run
            self.con.execute(
                "INSERT OR REPLACE INTO er_llm_decisions VALUES (?, ?, ?, ?, ?, ?, now())",
                [company.id, sanction.id, self.model, decision.is_match,
                 decision.confidence, decision.reasoning],
            )
        return decision


def decide(
    candidate: Candidate,
    thresholds: Thresholds,
    adjudicator: Adjudicator | None,
    company: Party,
    sanction: Party,
) -> tuple[str, str, float, str | None]:
    """Returns (decision, method, confidence, reasoning)."""
    if candidate.score >= thresholds.auto_match and not candidate.generic:
        # Ontology invariant: only identifier matches carry confidence 1.0.
        return MATCH, "FUZZY_NAME", min(candidate.score, 0.99), None
    if adjudicator is None:
        return REVIEW, "NEEDS_REVIEW", candidate.score, None
    verdict = adjudicator.adjudicate(company, sanction, candidate)
    if verdict is None:
        return REVIEW, "NEEDS_REVIEW", candidate.score, None
    decision = MATCH if verdict.is_match else NO_MATCH
    return decision, "LLM_ADJUDICATED", min(verdict.confidence, 0.99), verdict.reasoning


def resolve(
    con: duckdb.DuckDBPyConnection,
    adjudicator: Adjudicator | None = None,
    cfg: MatcherConfig | None = None,
    thresholds: Thresholds | None = None,
) -> dict[str, int]:
    thresholds = thresholds or Thresholds()
    companies = load_company_parties(con)
    sanctions = load_sanction_parties(con)
    by_company = {p.id: p for p in companies}
    by_sanction = {p.id: p for p in sanctions}
    placeholders = ", ".join("?" * len(IDENTIFIER_METHODS))
    # Companies already linked by an identifier are resolved; fuzzy matching would only add a
    # weaker, possibly wrong second link.
    resolved = {
        lei for (lei,) in con.execute(
            f"SELECT DISTINCT lei FROM company_sanction_matches "
            f"WHERE match_method IN ({placeholders})", list(IDENTIFIER_METHODS),
        ).fetchall()
    }
    if adjudicator is not None:
        adjudicator = CachedAdjudicator(adjudicator, con)

    matcher = Matcher(sanctions, cfg)
    unresolved = [c for c in companies if c.id not in resolved]
    rows = []
    for cand in best_candidates(unresolved, matcher, thresholds.review):
        decision, method, confidence, reasoning = decide(
            cand, thresholds, adjudicator, by_company[cand.left_id], by_sanction[cand.right_id]
        )
        rows.append([cand.left_id, cand.right_id, cand.score, cand.name_score,
                     cand.country_relation, cand.left_name, cand.right_name,
                     decision, method, confidence, reasoning])

    con.execute("""
        CREATE OR REPLACE TABLE er_candidates (
            lei VARCHAR, sanction_id VARCHAR, score DOUBLE, name_score DOUBLE,
            country_relation VARCHAR, company_name VARCHAR, sanction_name VARCHAR,
            decision VARCHAR, method VARCHAR, confidence DOUBLE, reasoning VARCHAR
        )
    """)
    if rows:
        con.executemany(f"INSERT INTO er_candidates VALUES ({', '.join('?' * 11)})", rows)

    # Rebuild the fuzzy part of the match table; identifier matches are owned by the warehouse.
    con.execute(
        f"DELETE FROM company_sanction_matches WHERE match_method NOT IN ({placeholders})",
        list(IDENTIFIER_METHODS),
    )
    matches = [
        [r[0], r[1], r[8], r[9],
         json.dumps({"company_name": r[5], "sanction_name": r[6], "score": round(r[2], 4),
                     "country_relation": r[4], "reasoning": r[10]}, ensure_ascii=False)]
        for r in rows if r[7] == MATCH
    ]
    if matches:
        con.executemany("INSERT INTO company_sanction_matches VALUES (?, ?, ?, ?, ?)", matches)

    counts = {
        "candidates": len(rows),
        "fuzzy_matches": sum(r[7] == MATCH and r[8] == "FUZZY_NAME" for r in rows),
        "llm_matches": sum(r[7] == MATCH and r[8] == "LLM_ADJUDICATED" for r in rows),
        "llm_rejections": sum(r[7] == NO_MATCH for r in rows),
        "needs_review": sum(r[7] == REVIEW for r in rows),
    }
    if isinstance(adjudicator, CachedAdjudicator):
        counts["llm_calls"] = adjudicator.calls
        counts["llm_cache_hits"] = adjudicator.hits
    log.info("entity resolution: %s", counts)
    return counts
