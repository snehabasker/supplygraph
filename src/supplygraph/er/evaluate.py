"""Entity-resolution evaluation against an objective gold set.

Labels come from identifiers, never from names: a GLEIF company is a positive for a sanctions
record if that record lists the company's LEI, or its local registration number (e.g. Russian
OGRN) in the same country — exactly the deterministic logic in `warehouse.build_matches`.
The fuzzy matcher never sees identifiers, so we measure how well names + countries alone
recover those links.

Population: (a) every GLEIF entity whose LEI appears on a sanctions record, plus (b) entities
sampled from high-exposure jurisdictions (RU, AE, CY, BY) and ordinary supply-chain countries.
Sample (b) is labelled by the same identifier rule, so sanctioned companies whose LEI is not
listed (e.g. Mosenergo, matched by OGRN) become positives instead of fake "false alarms".

- Companies are split dev/test by a hash of the LEI (stable, ~50/50).
- Thresholds are tuned on dev only; headline numbers are reported on test.
- Residual label noise: a company can be sanctioned with *no* identifier on its sanctions
  record. Such cases show up as false alarms and are listed in the report for spot-checking.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import duckdb

from supplygraph.er.adjudicate import Adjudicator
from supplygraph.er.matcher import Candidate, Matcher, MatcherConfig, Party
from supplygraph.er.resolve import (
    MATCH,
    CachedAdjudicator,
    Thresholds,
    company_party,
    decide,
    load_sanction_parties,
)
from supplygraph.ingest.gleif import GleifClient, compact_record, is_valid_lei, parse_record
from supplygraph.warehouse import (
    ORG_SCHEMAS,
    build_matches,
    company_rows,
    create_companies_table,
)

# Negative sampling plan: (country, pages of 100). High-exposure jurisdictions are
# over-represented on purpose — they produce the hardest look-alike names.
NEGATIVE_PLAN = (("RU", 2), ("AE", 1), ("CY", 1), ("BY", 1), ("US", 1), ("DE", 1), ("CN", 1))
NEGATIVE_PAGE_OFFSET = 7  # skip the first pages (oldest registrations) for a less biased sample


def split_of(lei: str) -> str:
    return "dev" if int(hashlib.sha1(lei.encode()).hexdigest(), 16) % 2 == 0 else "test"


def sanctioned_leis(con: duckdb.DuckDBPyConnection) -> dict[str, set[str]]:
    """LEI -> sanctions ids, for organisation records that carry a valid LEI."""
    placeholders = ", ".join("?" * len(ORG_SCHEMAS))
    rows = con.execute(
        f"""SELECT DISTINCT id, trim(unnest(string_split(identifiers, ';')))
            FROM sanctions
            WHERE identifiers IS NOT NULL AND schema IN ({placeholders})""",
        list(ORG_SCHEMAS),
    ).fetchall()
    out: dict[str, set[str]] = {}
    for sid, ident in rows:
        if is_valid_lei(ident):
            out.setdefault(ident, set()).add(sid)
    return out


def identifier_labels(con: duckdb.DuckDBPyConnection, records: list[dict]) -> dict[str, set[str]]:
    """LEI -> sanctions ids linked by identifier, using the production matching SQL."""
    create_companies_table(con, "eval_companies", company_rows(records), temp=True)
    build_matches(con, "eval_companies", "eval_identifier_matches", temp=True)
    labels: dict[str, set[str]] = {r["id"]: set() for r in records}
    for lei, sid in con.execute(
        "SELECT lei, sanction_id FROM eval_identifier_matches"
    ).fetchall():
        labels[lei].add(sid)
    con.execute("DROP TABLE eval_identifier_matches")
    con.execute("DROP TABLE eval_companies")
    return labels


def build_eval_set(con: duckdb.DuckDBPyConnection, client: GleifClient, out_dir: Path) -> dict:
    records: dict[str, dict] = {}
    for record in client.get_records(sorted(sanctioned_leis(con))):
        records[record["id"]] = record
    for country, pages in NEGATIVE_PLAN:
        for page in range(pages):
            for record in client.list_by_country(country, NEGATIVE_PAGE_OFFSET + page):
                records.setdefault(record["id"], record)
    labels = identifier_labels(con, list(records.values()))
    write_eval_set(out_dir, list(records.values()), labels)
    positives = sum(1 for ids in labels.values() if ids)
    return {"companies": len(labels), "positives": positives,
            "negatives": len(labels) - positives}


def write_eval_set(out_dir: Path, records: list[dict], labels: dict[str, set[str]]) -> None:
    """gold_pairs.csv has one row per (lei, sanction_id); negatives have an empty sanction_id."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for lei, ids in labels.items():
        rows += [(lei, sid, split_of(lei)) for sid in sorted(ids)] or [(lei, "", split_of(lei))]
    with (out_dir / "gold_pairs.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lei", "sanction_id", "split"])
        writer.writerows(sorted(rows))
    with (out_dir / "gleif_records.jsonl").open("w", encoding="utf-8") as fh:
        for record in sorted(records, key=lambda r: r["id"]):
            fh.write(json.dumps(compact_record(record), ensure_ascii=False, sort_keys=True) + "\n")


def load_eval_set(eval_dir: Path) -> tuple[dict[str, Party], dict[str, set[str]], dict[str, str]]:
    """Returns (company parties by LEI, gold sanction ids by LEI — empty for negatives,
    split by LEI)."""
    parties = {}
    with (eval_dir / "gleif_records.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = parse_record(json.loads(line))
            parties[row["lei"]] = company_party(
                row["lei"], row["legal_name"], row["other_names"],
                row["legal_country"], row["hq_country"],
            )
    gold: dict[str, set[str]] = {}
    splits: dict[str, str] = {}
    with (eval_dir / "gold_pairs.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ids = gold.setdefault(row["lei"], set())
            if row["sanction_id"]:
                ids.add(row["sanction_id"])
            splits[row["lei"]] = row["split"]
    return parties, gold, splits


@dataclass(frozen=True)
class Metrics:
    threshold: float
    positives: int
    negatives: int
    tp: int
    wrong_record: int  # sanctioned company linked to the wrong sanctions record
    false_alarms: int  # clean company flagged as sanctioned

    @property
    def fp(self) -> int:
        return self.wrong_record + self.false_alarms

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / self.positives if self.positives else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.negatives if self.negatives else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "precision": round(self.precision, 4),
                "recall": round(self.recall, 4), "f1": round(self.f1, 4),
                "false_alarm_rate": round(self.false_alarm_rate, 4)}


def metrics_at(
    top1: dict[str, Candidate | None],
    gold: dict[str, set[str]],
    threshold: float,
    accept: Callable[[Candidate], bool] | None = None,
) -> Metrics:
    """Each company predicts its top candidate if it clears `threshold` (and `accept`)."""
    tp = wrong = alarms = 0
    for lei, cand in top1.items():
        if cand is None or cand.score < threshold or (accept and not accept(cand)):
            continue
        truth = gold.get(lei, set())
        if not truth:
            alarms += 1
        elif cand.right_id in truth:
            tp += 1
        else:
            wrong += 1
    positives = sum(1 for lei in top1 if gold.get(lei))
    return Metrics(threshold, positives, len(top1) - positives, tp, wrong, alarms)


def top1_predictions(
    parties: dict[str, Party],
    sanctions: list[Party],
    gold: dict[str, set[str]],
    cfg: MatcherConfig,
    min_score: float,
) -> tuple[dict[str, Candidate | None], float]:
    """Top candidate per company, plus blocking recall over positives (is the gold record
    among the candidates at all? — an upper bound on achievable recall)."""
    matcher = Matcher(sanctions, cfg)
    top1, found, positives = {}, 0, 0
    for lei, party in parties.items():
        cands = matcher.candidates(party, min_score)
        top1[lei] = cands[0] if cands else None
        if gold.get(lei):
            positives += 1
            found += any(c.right_id in gold[lei] for c in cands)
    return top1, found / positives if positives else 0.0


def choose_thresholds(dev: list[Metrics], min_precision: float = 0.95) -> Thresholds:
    """auto_match: lowest threshold whose dev precision >= min_precision (most recall at that
    precision). review: best-F1 threshold below it, so the LLM/human band catches the rest."""
    precise = [m for m in dev if m.precision >= min_precision and m.tp > 0]
    auto = min(precise, key=lambda m: m.threshold).threshold if precise else 0.99
    below = [m for m in dev if m.threshold < auto]
    review = max(below, key=lambda m: (m.recall, m.f1)).threshold if below else auto - 0.05
    review = min(review, auto - 0.05)  # always leave a band for the LLM / human review
    return Thresholds(auto_match=round(auto, 2), review=round(max(review, 0.5), 2))


THRESHOLD_GRID = tuple(round(0.60 + 0.01 * i, 2) for i in range(40))


def evaluate(
    con: duckdb.DuckDBPyConnection,
    eval_dir: Path,
    cfg: MatcherConfig | None = None,
    adjudicator: Adjudicator | None = None,
) -> dict:
    cfg = cfg or MatcherConfig()
    parties, gold, splits = load_eval_set(eval_dir)
    sanctions = load_sanction_parties(con)
    live_ids = {p.id for p in sanctions}
    # Sanctions records can be delisted or merged after the eval set was built: drop positives
    # whose gold record is gone rather than counting them as misses.
    for lei in [lei for lei, ids in gold.items() if ids and not ids & live_ids]:
        del gold[lei]
    gold = {lei: ids & live_ids for lei, ids in gold.items()}
    parties = {lei: p for lei, p in parties.items() if lei in gold}
    by_sanction = {p.id: p for p in sanctions}

    def auto_ok(cand: Candidate) -> bool:  # same rule as resolve.decide
        return not cand.generic

    def run(config: MatcherConfig) -> dict:
        top1, blocking_recall = top1_predictions(parties, sanctions, gold, config, min_score=0.5)
        out: dict = {"blocking_recall": blocking_recall}
        for split in ("dev", "test"):
            sub = {lei: c for lei, c in top1.items() if splits[lei] == split}
            out[split] = {"top1": sub, "sweep": [metrics_at(sub, gold, t, auto_ok)
                                                 for t in THRESHOLD_GRID]}
        return out

    main = run(cfg)
    thresholds = choose_thresholds(main["dev"]["sweep"])
    test_top1 = main["test"]["top1"]

    def is_auto(c: Candidate | None) -> bool:
        return c is not None and c.score >= thresholds.auto_match and auto_ok(c)

    def count(split: str, positive: bool) -> int:
        return sum(1 for lei in parties if splits[lei] == split and bool(gold[lei]) == positive)

    review_band = [c for c in test_top1.values()
                   if c and c.score >= thresholds.review and not is_auto(c)]
    report: dict = {
        "companies": {s: {"positives": count(s, True), "negatives": count(s, False)}
                      for s in ("dev", "test")},
        "blocking_recall": round(main["blocking_recall"], 4),
        "chosen_thresholds_on_dev": asdict(thresholds),
        "test": {
            "auto_match_only":
                metrics_at(test_top1, gold, thresholds.auto_match, auto_ok).as_dict(),
            "if_review_band_all_accepted":
                metrics_at(test_top1, gold, thresholds.review).as_dict(),
            "review_queue": {
                "size": len(review_band),
                "true_matches_in_queue": sum(c.right_id in gold[c.left_id] for c in review_band),
            },
        },
        "test_false_alarms_at_auto": [
            {"lei": lei, "company": c.left_name, "matched": c.right_name,
             "sanction_id": c.right_id, "score": round(c.score, 3)}
            for lei, c in sorted(test_top1.items()) if is_auto(c) and not gold[lei]
        ],
        "test_misses_at_auto": [
            {"lei": lei, "company": parties[lei].names[0],
             "best_candidate": c.right_name if c else None,
             "score": round(c.score, 3) if c else None,
             "single_token": c.generic if c else None}
            for lei, c in sorted(test_top1.items())
            if gold[lei] and not (is_auto(c) and c.right_id in gold[lei])
        ],
        "dev_sweep": [m.as_dict() for m in main["dev"]["sweep"]],
    }

    # Ablations: what does each design choice buy? (dev-chosen thresholds, test split)
    ablations = {
        "allow_single_token_auto_match":
            metrics_at(test_top1, gold, thresholds.auto_match).as_dict(),
    }
    for name, variant in {
        "no_transliteration": replace(cfg, translit=False),
        "keep_legal_forms": replace(cfg, drop_legal_forms=False),
        "no_country_signal": replace(cfg, country_bonus=0.0, country_penalty=0.0),
    }.items():
        ablated = run(variant)["test"]["top1"]
        ablations[name] = metrics_at(ablated, gold, thresholds.auto_match, auto_ok).as_dict()
    report["test_ablations_at_auto_threshold"] = ablations

    if adjudicator is not None:
        cached = CachedAdjudicator(adjudicator, con)

        def accept(cand: Candidate) -> bool:
            decision, *_ = decide(cand, thresholds, cached, parties[cand.left_id],
                                  by_sanction[cand.right_id])
            return decision == MATCH

        hybrid = metrics_at(test_top1, gold, thresholds.review, accept=accept)
        report["test"]["hybrid_with_llm"] = {
            **hybrid.as_dict(), "model": cached.model,
            "llm_calls": cached.calls, "llm_cache_hits": cached.hits,
        }
    return report
