"""Multi-hop benchmark (graph_bench.py): gold semantics, question set, scoring.

World (supply links point supplier -> customer; OWNED_BY child -> parent):

    T2 --lead--> SUP --lead--> CUST <--asserted-- SANC (sanctioned)      SUB --OWNED_BY--> SANC
    CAND (sanctions candidate) --OWNED_BY--> HOLD
"""

from __future__ import annotations

import duckdb
import pytest

from supplygraph import graph_bench as gb

T2, SUP, CUST, SANC, SUB, CAND, HOLD, FAR = (
    "T2", "SUP", "CUST", "SANC", "SUB", "CAND", "HOLD", "FAR")


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE companies (lei VARCHAR, legal_name VARCHAR, other_names VARCHAR, "
              "legal_country VARCHAR, entity_status VARCHAR)")
    c.executemany("INSERT INTO companies VALUES (?, ?, NULL, 'US', 'ACTIVE')",
                  [(x, f"{x.title()} Corp") for x in (T2, SUP, CUST, SANC, SUB, CAND, HOLD,
                                                      FAR)])
    c.execute("CREATE TABLE ownership (child_lei VARCHAR, parent_lei VARCHAR, "
              "relationship_type VARCHAR)")
    c.executemany("INSERT INTO ownership VALUES (?, ?, 'DIRECT')", [(SUB, SANC), (CAND, HOLD)])
    c.execute("CREATE TABLE supply_links (supplier_lei VARCHAR, customer_lei VARCHAR, "
              "link_type VARCHAR, evidence VARCHAR)")
    c.executemany("INSERT INTO supply_links VALUES (?, ?, ?, ?)", [
        (T2, SUP, "POSSIBLE_SUPPLIER", "T2 makes parts"),
        (SUP, CUST, "POSSIBLE_SUPPLIER", "Sup ships modules"),
        (SANC, CUST, "SUPPLIES", "We buy crude from Sanc"),
        (FAR, HOLD, "POSSIBLE_SUPPLIER", "Far supplies Hold"),
    ])
    c.execute("CREATE TABLE sanctions (id VARCHAR, name VARCHAR)")
    c.executemany("INSERT INTO sanctions VALUES (?, ?)", [("S-1", "Sanc"), ("S-2", "Cand")])
    c.execute("CREATE TABLE company_sanction_matches (lei VARCHAR, sanction_id VARCHAR)")
    c.execute("INSERT INTO company_sanction_matches VALUES (?, 'S-1')", [SANC])
    c.execute("CREATE TABLE er_candidates (lei VARCHAR, sanction_id VARCHAR, "
              "sanction_name VARCHAR, decision VARCHAR)")
    c.executemany("INSERT INTO er_candidates VALUES (?, ?, ?, ?)",
                  [(CAND, "S-2", "Cand", "review"), (FAR, "S-9", "Nope", "no_match")])
    yield c
    c.close()


def gold(con, family, lei):
    return gb.gold_answer(gb.WarehouseGraph(con), family, lei)


def test_supplier_tiers_follow_supply_links_backwards_to_the_depth(con) -> None:
    assert gold(con, "supplier_tiers", CUST) == (
        sorted([f"lei:{SUP}", f"lei:{SANC}", f"lei:{T2}"]), 2)
    assert gold(con, "supplier_tiers", T2) == ([], 0)


def test_disruption_travels_to_customers_and_parents_not_to_subsidiaries(con) -> None:
    assert gold(con, "disruption_reach", T2) == ([f"lei:{CUST}", f"lei:{SUP}"], 2)
    # SANC's subsidiary SUB is not reached: disruption does not flow parent -> child.
    assert gold(con, "disruption_reach", SANC) == ([f"lei:{CUST}"], 1)
    assert gold(con, "disruption_reach", CAND) == ([f"lei:{HOLD}"], 1)


def test_sanctions_exposure_follows_ownership_both_ways_and_supply_forward(con) -> None:
    assert gold(con, "sanctions_exposure", SANC) == (["sanction:S-1"], 0)
    assert gold(con, "sanctions_exposure", SUB) == (["sanction:S-1"], 1)  # child of SANC
    assert gold(con, "sanctions_exposure", CUST) == (["sanction:S-1"], 1)  # SANC's customer
    assert gold(con, "sanctions_exposure", HOLD) == (["sanction:S-2"], 1)  # parent of CAND


def test_suppliers_of_an_exposed_company_are_not_exposed(con) -> None:
    # SUP -> CUST <- SANC: two hops apart, but risk does not flow customer -> supplier.
    assert gold(con, "sanctions_exposure", SUP) == ([], 0)
    assert gold(con, "sanctions_exposure", FAR) == ([], 0)  # supplies HOLD; same rule
    # A rejected ER candidate (decision != review) is not a sanctions link at all.
    assert "sanction:S-9" not in gold(con, "sanctions_exposure", FAR)[0]


def test_question_set_balances_exposed_and_hard_negative_companies(con) -> None:
    qs = gb.build_questions(con, {x: f"{x.title()} Corp" for x in
                                  (T2, SUP, CUST, SANC, SUB, CAND, HOLD, FAR)})
    by = {f: [q for q in qs if q["family"] == f] for f in gb.FAMILIES}
    assert {q["lei"] for q in by["supplier_tiers"]} == {SUP, CUST, HOLD}
    assert {q["lei"] for q in by["disruption_reach"]} == {T2, SUP, SANC, FAR}
    exposure = by["sanctions_exposure"]
    positives = [q for q in exposure if q["gold"]]
    negatives = [q for q in exposure if not q["gold"]]
    assert {q["lei"] for q in positives} == {SANC, SUB, CUST, CAND, HOLD}
    # As many negatives as positives where the graph has them (here only 3 are unexposed).
    assert {q["lei"] for q in negatives} == {SUP, FAR, T2}
    assert all("Corp" in q["question"] for q in qs)


def test_hard_negatives_are_the_unexposed_companies_nearest_a_sanctioned_one(con) -> None:
    con.execute("DELETE FROM er_candidates")  # positives: SANC, SUB, CUST -> 3 negatives
    con.execute("INSERT INTO companies VALUES ('LONE', 'Lone Corp', NULL, 'US', 'ACTIVE')")
    con.execute("INSERT INTO supply_links VALUES ('LONE', 'FAR', 'POSSIBLE_SUPPLIER', 'x')")
    qs = gb.build_questions(con, {x: x for x in (T2, SUP, CUST, SANC, SUB, CAND, HOLD, FAR,
                                                 "LONE")})
    negatives = {q["lei"] for q in qs if q["family"] == "sanctions_exposure" and not q["gold"]}
    # SUP is 2 undirected hops from SANC, T2 3; FAR/HOLD/CAND/LONE are nowhere near it.
    assert len(negatives) == 3 and {SUP, T2} <= negatives and "LONE" not in negatives


def test_stale_gold_is_detected(con) -> None:
    qs = gb.build_questions(con, {x: x for x in (T2, SUP, CUST, SANC, SUB, CAND, HOLD, FAR)})
    assert gb.stale_gold(qs, con) == []
    con.execute("DELETE FROM supply_links WHERE supplier_lei = ?", [T2])
    stale = gb.stale_gold(qs, con)
    assert f"supplier_tiers-{CUST}" in stale and f"disruption_reach-{T2}" in stale


def test_question_file_round_trips(con, tmp_path) -> None:
    qs = gb.build_questions(con, {x: x for x in (T2, SUP, CUST, SANC, SUB, CAND, HOLD, FAR)})
    path = tmp_path / "graph_bench" / "questions.jsonl"
    gb.write_questions(qs, path)
    assert gb.read_questions(path) == qs


@pytest.mark.parametrize(("gold_refs", "predicted", "expected"), [
    (["a", "b"], {"a", "x"}, {"recall": 0.5, "precision": 0.5, "exact": False,
                              "false_alarm": False}),
    (["a"], set(), {"recall": 0.0, "precision": None, "exact": False, "false_alarm": False}),
    ([], set(), {"recall": None, "precision": None, "exact": True, "false_alarm": False}),
    ([], {"x"}, {"recall": None, "precision": 0.0, "exact": False, "false_alarm": True}),
])
def test_score_leaves_undefined_metrics_undefined(gold_refs, predicted, expected) -> None:
    got = gb.score(gold_refs, predicted)
    assert {k: got[k] for k in expected} == expected


def test_summary_averages_only_where_a_metric_is_defined() -> None:
    def m(recall, precision, exact, fa):
        return {"recall": recall, "precision": precision, "exact": exact, "false_alarm": fa,
                "tokens": 10, "ms": 1.0}

    rows = [{"gold": 2, **{k: m(1.0, 0.5, False, False) for k in gb.METHODS}},
            {"gold": 0, **{k: m(None, None, True, False) for k in gb.METHODS}},
            {"gold": 0, **{k: m(None, 0.0, False, True) for k in gb.METHODS}}]
    s = gb.summarize(rows)["vector_rag"]
    assert (s["recall"], s["precision"], s["exact"], s["false_alarm"]) == (1.0, 0.25, 0.333,
                                                                           0.5)


def test_vector_index_ranks_the_chunk_that_shares_the_rare_words(con) -> None:
    chunks = gb.vector_chunks(con)
    hits = gb.VectorIndex(chunks).search("Which sanctions record does Cand Corp match?", 3)
    assert "sanction:S-2" in hits[0]["refs"]
    assert gb.VectorIndex(chunks).search("zzz qqq", 3) == []  # nothing in common


class FakeCypher:
    """Stands in for CypherTools: fixed tool answers and a fixed neighbourhood."""

    def __init__(self) -> None:
        self.handlers = {
            "supplier_tree": lambda lei, depth, limit: {
                "ref": f"lei:{lei}", "suppliers": [{"ref": f"lei:{SUP}"}, {"ref": f"lei:{T2}"}]},
            "disruption_reach": lambda lei, depth, limit: {"ref": f"lei:{lei}", "affected": []},
            "sanctions_exposure": lambda lei, max_hops: {"ref": f"lei:{lei}", "exposures": []},
        }

    def read(self, query: str, **params) -> list[dict]:
        return [{"a_lei": SUP, "a_id": None, "a_name": "Sup", "type": "POSSIBLE_SUPPLIER",
                 "b_lei": CUST, "b_id": None, "b_name": "Cust"}]


def test_evaluate_scores_each_method_on_its_own_context(con) -> None:
    q = {"id": "q1", "family": "supplier_tiers", "lei": CUST, "name": "Cust Corp",
         "question": "Which companies supply Cust Corp, up to two tiers?",
         "gold": sorted([f"lei:{SUP}", f"lei:{SANC}", f"lei:{T2}"]), "hops_needed": 2}
    report = gb.evaluate([q], con, FakeCypher(), k=3, radius=2)
    row = report["rows"][0]
    assert row["linked"] is True
    assert row["cypher_tools"]["recall"] == pytest.approx(2 / 3)
    assert row["cypher_tools"]["precision"] == 1.0
    assert row["graph_rag"]["recall"] == pytest.approx(1 / 3)  # only SUP besides the seed
    assert report["multi_hop"]["questions"] == 1
    assert set(report["summary"]) >= {"questions", *gb.METHODS}
