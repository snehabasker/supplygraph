"""Entity resolution: normalization, matching, decisions, LLM adjudication, evaluation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from conftest import make_lei, make_record, now_iso, write_raw_data
from supplygraph import quality, warehouse
from supplygraph.er import evaluate as ev
from supplygraph.er.adjudicate import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    ClaudeAdjudicator,
    Decision,
    MatchVerdict,
)
from supplygraph.er.matcher import Candidate, Matcher, MatcherConfig, Party, name_similarity
from supplygraph.er.normalize import normalize
from supplygraph.er.resolve import Thresholds, resolve
from supplygraph.graph import fetch_dicts
from supplygraph.ingest.gleif import compact_record, parse_record

MOSENERGO = make_lei("TESTER000000000001")
ATLANT = make_lei("TESTER000000000002")
BORDER = make_lei("TESTER000000000003")
REGCO = make_lei("TESTER000000000004")
CLEAN = make_lei("TESTER000000000005")


# --- normalization ------------------------------------------------------------------------

def test_normalize_transliterates_and_drops_legal_forms() -> None:
    assert normalize('Публичное акционерное общество "Мосэнерго"') == "mosenergo"
    assert normalize("PJSC Mosenergo") == "mosenergo"
    assert normalize("Nestlé S.A.") == "nestle"
    assert normalize("Limited Liability Company \"AG Capital\"") == "capital"
    # Regression: long legal forms made Siemens look similar to "EK-Company AG".
    assert normalize("Siemens Aktiengesellschaft") == "siemens"
    assert normalize("Robert Bosch Gesellschaft mit beschränkter Haftung") == "robert bosch"


def test_normalize_can_disable_steps_for_ablations() -> None:
    assert normalize("Мосэнерго", translit=False) == ""  # Cyrillic is dropped, not matched
    assert "pjsc" in normalize("PJSC Mosenergo", drop_legal_forms=False)


# --- matching -----------------------------------------------------------------------------

def test_single_token_names_do_not_get_subset_credit() -> None:
    """Regression: 'AG Capital' scored 1.0 against 'RT-Capital' via token_set_ratio."""
    assert name_similarity("capital", "rt capital") < 0.9
    # Multi-token names still get partial subset credit — but an extra word ("trading") can
    # mean a different subsidiary, so it must stay well below the auto-match threshold.
    assert 0.8 < name_similarity("rosneft oil", "rosneft oil trading") < 0.9
    assert name_similarity("rosneft", "rosneft trading") < 0.7


def test_matcher_blocks_scores_and_flags_generic_names() -> None:
    sanctions = [
        Party("S1", ("PJSC Mosenergo",), frozenset({"RU"})),
        Party("S2", ("Atlant LLC",), frozenset({"RU"})),
    ]
    matcher = Matcher(sanctions)
    best = matcher.best(Party("C1", ("Публичное акционерное общество Мосэнерго",),
                                frozenset({"RU"})))
    assert best is not None and best.right_id == "S1" and best.score == 1.0
    assert best.generic  # "mosenergo" is a single token after normalization

    two_tokens = Matcher([Party("S3", ("Severgaz Bank JSC",), frozenset({"RU"}))])
    assert not two_tokens.best(Party("C2", ("Severgaz Bank",), frozenset({"RU"}))).generic


def test_country_mismatch_lowers_score() -> None:
    matcher = Matcher([Party("S1", ("Kama Trading House LLC",), frozenset({"RU"}))])
    same = matcher.best(Party("C", ("Kama Trading House",), frozenset({"RU"})))
    other = matcher.best(Party("C", ("Kama Trading House",), frozenset({"US"})))
    assert same.score > other.score
    assert (same.country_relation, other.country_relation) == ("match", "mismatch")


def test_very_common_tokens_are_not_used_for_blocking() -> None:
    sanctions = [Party(f"S{i}", (f"Bank {i}x",), frozenset()) for i in range(5)]
    matcher = Matcher(sanctions, MatcherConfig(max_token_frequency=3))
    assert matcher.candidates(Party("C", ("Bank",), frozenset())) == []


# --- production resolve -------------------------------------------------------------------

def er_records() -> dict[str, dict]:
    return {
        MOSENERGO: make_record(MOSENERGO, "ПАО энергетики и электрификации Мосэнерго", "RU"),
        ATLANT: make_record(ATLANT, 'ООО "Атлант"', "RU"),
        BORDER: make_record(BORDER, "Kama Trading House", "RU"),
        REGCO: make_record(REGCO, "Some Registered Plant", "RU", registered_as="1027700302420"),
        CLEAN: make_record(CLEAN, "Totally Unrelated Widgets", "US"),
    }


def er_sanctions() -> list[dict]:
    seen = now_iso()
    return [
        {"id": "S-MOS", "schema": "Company", "countries": "ru", "last_seen": seen,
         "name": "Public Joint Stock Company of Power and Electrification Mosenergo"},
        {"id": "S-ATL", "schema": "Company", "name": "Atlant LLC", "countries": "ru",
         "last_seen": seen},
        # Borderline on purpose: an extra *meaningful* word ("volga") keeps it below auto-match.
        # (Filler like "Group Holding" would be normalized away and match exactly.)
        {"id": "S-KAMA", "schema": "Company", "name": "Kama Trading House Volga",
         "countries": "ru", "last_seen": seen},
        {"id": "S-REG", "schema": "Company", "name": "Different Name Entirely",
         "countries": "ru", "identifiers": "1027700302420", "last_seen": seen},
    ]


@pytest.fixture
def er_con(tmp_path):
    write_raw_data(tmp_path, er_records(), set(), er_sanctions())
    con = warehouse.connect(":memory:")
    warehouse.build_all(con, tmp_path / "raw" / "gleif",
                        tmp_path / "raw" / "opensanctions" / "targets.simple.csv")
    yield con
    con.close()


def matches(con) -> dict[str, tuple[str, str]]:
    return {r["lei"]: (r["sanction_id"], r["match_method"]) for r in fetch_dicts(
        con, "SELECT lei, sanction_id, match_method FROM company_sanction_matches")}


def decisions(con) -> dict[str, str]:
    return dict(con.execute("SELECT lei, decision FROM er_candidates").fetchall())


class FakeAdjudicator:
    def __init__(self, verdicts: dict[str, Decision | None]) -> None:
        self.model = "fake-model"
        self.verdicts = verdicts
        self.calls: list[str] = []

    def adjudicate(self, company, sanction, candidate):
        self.calls.append(company.id)
        return self.verdicts.get(company.id)


def test_registration_number_links_only_within_the_same_country(er_con) -> None:
    assert matches(er_con)[REGCO] == ("S-REG", "REGISTRATION_NUMBER")


@pytest.mark.parametrize(("identifier", "countries"), [("1027700302420", "kz"), ("12345", "ru")])
def test_registration_number_rejects_other_country_or_short_numbers(
    tmp_path, identifier, countries
) -> None:
    recs = {REGCO: make_record(REGCO, "Some Plant", "RU", registered_as=identifier)}
    write_raw_data(tmp_path, recs, set(), [
        {"id": "S-REG", "schema": "Company", "name": "X", "countries": countries,
         "identifiers": identifier, "last_seen": now_iso()},
    ])
    con = warehouse.connect(":memory:")
    counts = warehouse.build_all(con, tmp_path / "raw" / "gleif",
                                 tmp_path / "raw" / "opensanctions" / "targets.simple.csv")
    assert counts["company_sanction_matches"] == 0


def test_resolve_without_llm_auto_matches_specific_names_and_queues_the_rest(er_con) -> None:
    counts = resolve(er_con, thresholds=Thresholds(auto_match=0.97, review=0.80))
    found, decided = matches(er_con), decisions(er_con)
    assert REGCO not in decided  # identifier-resolved companies skip fuzzy matching
    assert decided[ATLANT] == "review"  # exact but generic single-token name
    assert decided[BORDER] == "review"  # similar, but an extra meaningful word
    assert ATLANT not in found and BORDER not in found and CLEAN not in found
    assert counts["needs_review"] == 2 and counts["fuzzy_matches"] == 0
    # Every fuzzy match respects the ontology's confidence rule.
    assert not [r for r in quality.run_checks(er_con) if r.status == "fail"]


def test_multi_token_name_auto_matches_with_confidence_below_one(er_con) -> None:
    er_con.execute("UPDATE sanctions SET name = 'Kama Trading House' WHERE id = 'S-KAMA'")
    resolve(er_con)
    assert matches(er_con)[BORDER] == ("S-KAMA", "FUZZY_NAME")
    (confidence,) = er_con.execute(
        "SELECT confidence FROM company_sanction_matches WHERE lei = ?", [BORDER]).fetchone()
    assert 0.9 < confidence < 1.0


def test_resolve_is_idempotent(er_con) -> None:
    def snapshot():
        return sorted(er_con.execute("SELECT * FROM company_sanction_matches").fetchall())

    first_counts, first_rows = resolve(er_con), snapshot()
    assert (resolve(er_con), snapshot()) == (first_counts, first_rows)


def test_llm_decides_the_review_band_and_is_cached(er_con) -> None:
    fake = FakeAdjudicator({
        ATLANT: Decision(False, 0.2, "Generic name, no corroboration.", "fake-model"),
        BORDER: Decision(True, 0.9, "Same trading house, same country.", "fake-model"),
        MOSENERGO: Decision(True, 1.0, "Same utility; Cyrillic vs English name.", "fake-model"),
    })
    thresholds = Thresholds(auto_match=0.99, review=0.70)
    resolve(er_con, fake, thresholds=thresholds)

    assert sorted(fake.calls) == sorted([ATLANT, BORDER, MOSENERGO])
    assert decisions(er_con)[ATLANT] == "no_match"
    assert matches(er_con)[BORDER] == ("S-KAMA", "LLM_ADJUDICATED")
    assert matches(er_con)[MOSENERGO] == ("S-MOS", "LLM_ADJUDICATED")
    (confidence,) = er_con.execute(  # the LLM's 1.0 is capped: only identifiers are certain
        "SELECT confidence FROM company_sanction_matches WHERE lei = ?", [MOSENERGO]).fetchone()
    assert confidence == 0.99
    evidence = json.loads(er_con.execute(
        "SELECT evidence FROM company_sanction_matches WHERE lei = ?", [BORDER]).fetchone()[0])
    assert evidence["reasoning"] == "Same trading house, same country."

    calls_before = len(fake.calls)
    counts = resolve(er_con, fake, thresholds=thresholds)
    assert len(fake.calls) == calls_before  # all answers came from the DuckDB cache
    assert counts["llm_calls"] == 0 and counts["llm_cache_hits"] == 3


def test_llm_failure_sends_pair_to_review_and_is_retried_next_run(er_con) -> None:
    fake = FakeAdjudicator({})  # every call "fails" (returns None)
    thresholds = Thresholds(auto_match=0.99, review=0.70)
    resolve(er_con, fake, thresholds=thresholds)
    assert decisions(er_con)[BORDER] == "review"
    first_calls = len(fake.calls)
    resolve(er_con, fake, thresholds=thresholds)
    assert len(fake.calls) == 2 * first_calls  # failures are not cached


def test_quality_check_catches_fuzzy_match_claiming_certainty(er_con) -> None:
    resolve(er_con)
    er_con.execute("INSERT INTO company_sanction_matches VALUES (?, 'S-ATL', 'FUZZY_NAME', 1.0,"
                   " '{}')", [ATLANT])
    failed = {r.name for r in quality.run_checks(er_con) if r.status == "fail"}
    assert "matches_confidence_rule" in failed


# --- Claude adjudicator (fake client; no network) -----------------------------------------

class FakeMessages:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.kwargs: dict = {}

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def fake_client(outcome) -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(outcome)))


def candidate() -> Candidate:
    return Candidate("L", "S", 0.92, 0.92, "match", "Kama Trading House", "Kama TH", 3)


PARTY_L = Party("L", ("Kama Trading House",), frozenset({"RU"}))
PARTY_S = Party("S", ("Kama TH",), frozenset({"RU"}))


def _response(stop_reason="end_turn", parsed=None, model=DEFAULT_MODEL):
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed, model=model)


def _error(cls, status: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


def test_claude_adjudicator_returns_structured_decision_and_sends_fallbacks() -> None:
    client = fake_client(_response(parsed=MatchVerdict(
        same_entity=True, confidence=1.7, reasoning="Same name and country.")))
    decision = ClaudeAdjudicator(client=client).adjudicate(PARTY_L, PARTY_S, candidate())
    assert decision == Decision(True, 1.0, "Same name and country.", DEFAULT_MODEL)  # clamped
    sent = client.beta.messages.kwargs
    assert sent["model"] == DEFAULT_MODEL
    assert sent["output_format"] is MatchVerdict
    assert sent["fallbacks"] == "default" and sent["betas"] == [FALLBACK_BETA]
    payload = sent["messages"][0]["content"]
    assert "Kama Trading House" in payload and "untrusted" in sent["system"]


@pytest.mark.parametrize("outcome", [
    _response(stop_reason="refusal"),
    _response(parsed=None),
    _error(anthropic.RateLimitError, 429),
    _error(anthropic.InternalServerError, 500),
])
def test_claude_adjudicator_failures_return_none(outcome) -> None:
    assert ClaudeAdjudicator(client=fake_client(outcome)).adjudicate(
        PARTY_L, PARTY_S, candidate()) is None


def test_claude_adjudicator_raises_on_misconfiguration() -> None:
    client = fake_client(_error(anthropic.AuthenticationError, 401))
    with pytest.raises(anthropic.AuthenticationError):
        ClaudeAdjudicator(client=client).adjudicate(PARTY_L, PARTY_S, candidate())


# --- evaluation ---------------------------------------------------------------------------

def _cand(left: str, right: str, score: float, tokens: int = 2) -> Candidate:
    return Candidate(left, right, score, score, "match", left, right, tokens)


def test_metrics_separate_wrong_record_from_false_alarm() -> None:
    gold = {"A": {"SA"}, "B": {"SB"}, "C": set(), "D": {"SD"}}
    top1 = {"A": _cand("A", "SA", 0.99), "B": _cand("B", "SX", 0.99),
            "C": _cand("C", "SC", 0.99), "D": _cand("D", "SD", 0.80)}
    m = ev.metrics_at(top1, gold, 0.95)
    assert (m.tp, m.wrong_record, m.false_alarms, m.positives, m.negatives) == (1, 1, 1, 3, 1)
    assert m.precision == pytest.approx(1 / 3) and m.recall == pytest.approx(1 / 3)
    assert m.false_alarm_rate == 1.0


def test_choose_thresholds_prefers_recall_at_target_precision() -> None:
    sweep = [ev.Metrics(t, 10, 10, tp, 0, fa) for t, tp, fa in
             [(0.80, 9, 5), (0.90, 8, 0), (0.95, 7, 0)]]
    chosen = ev.choose_thresholds(sweep)
    assert chosen.auto_match == 0.90
    assert chosen.review <= chosen.auto_match - 0.05


def test_evaluate_end_to_end_on_tiny_gold_set(er_con, tmp_path) -> None:
    records = list(er_records().values())
    labels = ev.identifier_labels(er_con, records)
    assert labels[REGCO] == {"S-REG"} and labels[CLEAN] == set()
    ev.write_eval_set(tmp_path / "eval", records, labels)
    report = ev.evaluate(er_con, tmp_path / "eval")
    assert set(report["test"]) >= {"auto_match_only", "review_queue"}
    assert set(report["test_ablations_at_auto_threshold"]) >= {
        "no_transliteration", "keep_legal_forms", "no_country_signal",
        "allow_single_token_auto_match"}
    # Labels never leak into the warehouse: the temp tables are gone.
    assert not warehouse.table_exists(er_con, "eval_companies")


def test_compact_record_preserves_parsed_fields() -> None:
    record = make_record(REGCO, "Some Plant", "RU", other_names=["Alt"], registered_as="123456")
    record["attributes"]["entity"]["unusedField"] = {"big": "x" * 1000}
    assert parse_record(compact_record(record)) == parse_record(record)
