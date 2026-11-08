"""Risk propagation, scenarios, sensitivity and supplier link prediction."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta

import pytest

from conftest import ROSNEFT, make_lei, make_record, news_event, write_jsonl
from supplygraph import pipeline, quality, warehouse
from supplygraph.risk import engine, linkpred
from supplygraph.risk.engine import (
    NEWS,
    SANCTIONED,
    SCENARIO,
    Edge,
    RiskConfig,
    Signal,
    kendall_tau,
    noisy_or,
    propagate,
)

CFG = RiskConfig()


def sig(lei: str, value: float = 1.0, kind: str = SANCTIONED) -> Signal:
    return Signal(lei, kind, value, "test")


def owned(child: str, parent: str, rel: str = "DIRECT") -> list[Edge]:
    """The two directed risk edges the engine derives from one OWNED_BY link."""
    return [Edge(parent, child, f"OWNED_BY:{rel}"), Edge(child, parent, "PARENT_OF")]


# --- propagation properties -------------------------------------------------------------

def test_noisy_or_is_bounded_and_monotone() -> None:
    assert noisy_or([]) == 0.0
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert noisy_or([1.0, 0.3]) == 1.0
    assert noisy_or([2.0, -1.0]) == 1.0  # clipped, never outside [0, 1]


def test_subsidiary_inherits_from_sanctioned_parent_with_explanation() -> None:
    result = propagate([sig("P")], owned("C", "P") + owned("G", "C"), CFG)
    assert result.scores["P"] == 1.0 and result.own["P"] == 1.0
    assert result.scores["C"] == pytest.approx(0.8)
    assert result.scores["G"] == pytest.approx(0.64)  # two hops: 0.8 * 0.8
    [driver] = result.drivers("G")
    assert driver.path == ("P", "C", "G")
    assert driver.relations == ("OWNED_BY:DIRECT", "OWNED_BY:DIRECT")
    assert result.own["G"] == 0.0


def test_strongest_path_wins_and_cycles_do_not_amplify() -> None:
    edges = (owned("C", "P", "ULTIMATE")  # direct weak link 0.6 ...
             + owned("M", "P") + owned("C", "M"))  # ... vs two strong hops 0.8 * 0.8 = 0.64
    result = propagate([sig("P")], edges, CFG)
    assert result.scores["C"] == pytest.approx(0.64)
    assert result.drivers("C")[0].path == ("P", "M", "C")
    # A supply cycle back to the source adds nothing: one signal, one (best) path.
    looped = propagate([sig("P")], [*edges, Edge("C", "P", "SUPPLIES")], CFG)
    assert looped.scores == result.scores


def test_hop_limit_and_minimum_contribution() -> None:
    chain = [Edge(str(i), str(i + 1), "SUPPLIES") for i in range(10)]
    result = propagate([sig("0")], chain, RiskConfig(max_hops=3))
    assert set(result.scores) == {"0", "1", "2", "3"}
    weak = propagate([sig("0", 0.03)], chain, CFG)  # 0.03 * 0.5 = 0.015 > 0.01, then stops
    assert set(weak.scores) == {"0", "1"}


def test_disruptions_follow_supply_not_parent_to_child() -> None:
    # Regression: news about a parent spread to every subsidiary on live data.
    edges = [*owned("SUB", "PARENT"), Edge("PARENT", "CUSTOMER", "SUPPLIES")]
    news = propagate([sig("PARENT", 0.2, NEWS)], edges, CFG)
    assert "SUB" not in news.scores
    assert news.scores["CUSTOMER"] == pytest.approx(0.1)
    upward = propagate([sig("SUB", 1.0, SCENARIO)], edges, CFG)  # a subsidiary's fab fire
    assert upward.scores["PARENT"] == pytest.approx(CFG.child_to_parent)
    sanction = propagate([sig("PARENT")], edges, CFG)
    assert sanction.scores["SUB"] == pytest.approx(0.8)  # sanctions do follow ownership


def test_adding_signals_or_links_never_lowers_a_score() -> None:
    rng = random.Random(7)
    nodes = [f"N{i}" for i in range(12)]
    relations = ["OWNED_BY:DIRECT", "PARENT_OF", "SUPPLIES", "POSSIBLE_SUPPLIER"]
    for _ in range(50):
        edges = [Edge(rng.choice(nodes), rng.choice(nodes), rng.choice(relations))
                 for _ in range(20)]
        signals = [sig(rng.choice(nodes), rng.random(), rng.choice([SANCTIONED, NEWS]))
                   for _ in range(3)]
        base = propagate(signals, edges, CFG, nodes)
        more_signals = propagate([*signals, sig(rng.choice(nodes), rng.random())], edges, CFG,
                                 nodes)
        extra_edge = Edge(rng.choice(nodes), rng.choice(nodes), "SUPPLIES")
        more_edges = propagate(signals, [*edges, extra_edge], CFG, nodes)
        for n in nodes:
            assert 0.0 <= base.scores[n] <= 1.0
            assert more_signals.scores[n] >= base.scores[n] - 1e-12
            assert more_edges.scores[n] >= base.scores[n] - 1e-12


def test_kendall_tau() -> None:
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert kendall_tau(["a", "b", "x"], ["b", "a", "y"]) == -1.0  # only shared items count


# --- warehouse integration --------------------------------------------------------------

CHILD = make_lei("TESTSG0000000RCHLD")
FOUNDRY = make_lei("TESTSG0000000FOUND")
CHIPCO = make_lei("TESTSG0000000CHIPC")
LONER = make_lei("TESTSG0000000LONER")
NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


@pytest.fixture
def risk_con(raw_data_dir):
    """Rosneft (sanctioned by LEI) owns CHILD; FOUNDRY supplies CHIPCO; FOUNDRY is in the news."""
    raw = raw_data_dir / "raw"
    write_jsonl(raw / "gleif" / "suppliers" / "records.jsonl", [
        make_record(CHILD, "Rosneft Trading Test GmbH", "DE"),
        make_record(FOUNDRY, "Test Foundry Co., Ltd.", "TW"),
        make_record(CHIPCO, "Test Chip Inc.", "US"),
        make_record(LONER, "Unconnected Test Corp", "US"),
    ])
    write_jsonl(raw / "gleif" / "suppliers" / "relationships.jsonl", [
        {"child_lei": CHILD, "parent_lei": ROSNEFT, "relationship_type": "DIRECT"}])
    (raw / "gleif" / "seeds.json").write_text(json.dumps([FOUNDRY, CHIPCO]))
    from conftest import filing_row, mention_row
    write_jsonl(raw / "sec" / "filings.jsonl", [filing_row(CHIPCO, "77", "Test Chip")])
    write_jsonl(raw / "sec" / "supplier_mentions.jsonl", [
        mention_row(CHIPCO, "77", "Test Foundry Co., Ltd.", FOUNDRY, "LLM")])
    fresh = news_event("N-new", ["test foundry"], ["FIRE"])
    old = news_event("N-old", ["test foundry"], ["FIRE"])
    old["title"] = "Older story"
    old["published_at"] = (NOW - timedelta(days=3)).isoformat()
    fresh["published_at"] = NOW.isoformat()
    write_jsonl(raw / "gdelt" / "events.jsonl", [fresh, old])
    con = warehouse.connect(":memory:")
    warehouse.build_all(con, raw / "gleif", raw / "opensanctions" / "targets.simple.csv",
                        raw / "sec", raw / "gdelt")
    yield con
    con.close()


def test_signals_from_sanctions_and_decayed_capped_news(risk_con) -> None:
    signals = {s.lei: s for s in engine.load_signals(risk_con, CFG, NOW)}
    assert signals[ROSNEFT].kind == SANCTIONED and signals[ROSNEFT].value == 1.0
    # One fresh article (0.05) + one three days old, i.e. one half-life (0.025).
    assert signals[FOUNDRY].value == pytest.approx(0.075)
    capped = engine.load_signals(risk_con, RiskConfig(news_event=1.0, news_cap=0.2), NOW)
    assert {s.lei: s for s in capped}[FOUNDRY].value == 0.2


def test_scores_are_written_explained_and_pass_quality_checks(risk_con) -> None:
    counts = engine.write_scores(risk_con, engine.score(risk_con, CFG, now=NOW), CFG)
    assert counts["companies_scored"] == 7
    rows = {lei: (s, own, json.loads(d)) for lei, s, own, d in risk_con.execute(
        "SELECT lei, risk_score, own_risk, drivers FROM company_risk").fetchall()}
    score_, own, drivers = rows[CHILD]
    assert (score_, own) == (pytest.approx(0.8), 0.0)
    assert drivers[0]["path"] == ["Rosneft Oil Company", "Rosneft Trading Test GmbH"]
    assert rows[CHIPCO][0] == pytest.approx(0.0375)  # news at the foundry x 0.5 via SUPPLIES
    assert rows[LONER] == (0.0, 0.0, [])
    results = {r.name: r for r in quality.run_checks(risk_con)}
    for name in ("risk_scores_bounded", "risk_scores_explained", "risk_covers_companies"):
        assert results[name].status == "pass", name


@pytest.mark.parametrize("sql", [
    "UPDATE company_risk SET risk_score = 1.5 WHERE risk_score > 0.5",
    "UPDATE company_risk SET drivers = '[]' WHERE risk_score > 0",
    "DELETE FROM company_risk WHERE risk_score = 0",
])
def test_risk_checks_catch_corruption(risk_con, sql) -> None:
    engine.write_scores(risk_con, engine.score(risk_con, CFG, now=NOW), CFG)
    risk_con.execute(sql)
    assert quality.has_blocking_failures(quality.run_checks(risk_con))


def test_scenario_reports_who_is_hit_and_how(risk_con) -> None:
    impacted = engine.scenario(risk_con, FOUNDRY, 1.0, CFG, now=NOW)
    assert [r["lei"] for r in impacted] == [CHIPCO]
    [hit] = impacted
    assert hit["path"] == ["Test Foundry Co., Ltd.", "Test Chip Inc."]
    assert hit["relations"] == ["SUPPLIES"]
    assert hit["after"] == pytest.approx(noisy_or([0.0375, 0.5]), abs=1e-4)
    assert engine.scenario(risk_con, LONER, 1.0, CFG, now=NOW) == []


def test_sensitivity_reports_stability(risk_con) -> None:
    report = engine.sensitivity(risk_con, CFG, now=NOW)
    assert report["baseline_at_risk"] == 4  # Rosneft, its subsidiary, the foundry, its customer
    assert len(report["variations"]) == 2 * len(engine.TUNABLE)
    assert 0.0 <= report["min_top_k_overlap"] <= 1.0


def test_find_company_by_lei_name_or_filing_alias(risk_con) -> None:
    assert pipeline.find_company(risk_con, CHIPCO)[0] == CHIPCO
    assert pipeline.find_company(risk_con, "test chip")[0] == CHIPCO
    with pytest.raises(ValueError, match="exactly one"):
        pipeline.find_company(risk_con, "Nonexistent Holdings")


# --- link prediction --------------------------------------------------------------------

def test_rank_of_shares_ties_instead_of_breaking_them_luckily() -> None:
    scores = {"a": 2.0, "b": 1.0, "c": 1.0, "d": 1.0}
    assert linkpred.rank_of("a", scores, list(scores)) == 1
    assert linkpred.rank_of("c", scores, list(scores)) == 3  # tied for 2nd-4th
    assert linkpred.rank_of("zz", scores, [*list(scores), "zz"]) == 5


def test_leave_one_out_random_baseline_is_exact_and_popularity_beats_it() -> None:
    # Hub "T" supplies everyone; each customer also has one niche supplier.
    links = [("T", c) for c in "ABCD"] + [("n" + c, c) for c in "ABCD"]
    report = linkpred.leave_one_out(links)
    assert report["queries"] == 8
    rand, pop = report["methods"]["random"], report["methods"]["popularity"]
    # Every query has 4 candidates (5 suppliers minus the 1 still known): E[1/rank] = 25/48.
    assert rand["mrr"] == pytest.approx(25 / 48, abs=1e-3)
    assert rand["hit@1"] == pytest.approx(0.25)
    assert pop["mrr"] > rand["mrr"]


def test_cooccurrence_finds_suppliers_that_travel_together() -> None:
    # Customers using "mem" also use "dram"; X uses only "mem".
    links = [("mem", c) for c in "ABCX"] + [("dram", c) for c in "ABC"] + [("other", "Z")]
    scores = linkpred.cooccurrence("X", links)
    assert scores["dram"] > scores["other"] == 0.0


def test_recommend_excludes_known_suppliers(risk_con) -> None:
    risk_con.execute("INSERT INTO supply_links (supplier_lei, customer_lei, link_type) "
                     f"VALUES ('{LONER}', '{ROSNEFT}', 'POSSIBLE_SUPPLIER')")
    recs = linkpred.recommend(risk_con, CHIPCO)
    assert FOUNDRY not in [s for s, _ in recs] and recs[0][0] == LONER
