"""Supplier extraction: filing text, aliases, extractors, grounding, LEI linking, SEC ingest.

Sentences are taken from the real 10-K / 20-F filings the gold set was built from.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import pytest

from conftest import make_lei, make_record
from supplygraph import warehouse
from supplygraph.extract import link, suppliers
from supplygraph.extract.evaluate import GoldRow, score
from supplygraph.extract.suppliers import (
    CachedClaudeExtractor,
    ClaudeExtractor,
    FilerContext,
    Mention,
    PassageRelations,
    Relation,
    RuleExtractor,
    is_grounded,
)
from supplygraph.extract.text import (
    Passage,
    aliases_excluding_filer,
    candidate_passages,
    html_to_text,
    is_self,
    learn_aliases,
    merge_aliases,
    org_names,
    passage_id,
    split_sentences,
)
from supplygraph.ingest import sec

NVIDIA_SUPPLY = (
    "We utilize foundries, such as Taiwan Semiconductor Manufacturing Company Limited, or TSMC, "
    "and Samsung Electronics Co., Ltd., or Samsung, to produce our semiconductor wafers. "
    "We purchase memory from SK Hynix Inc., Micron Technology, Inc., and Samsung. "
    "We engage with independent subcontractors and contract manufacturers such as Hon Hai "
    "Precision Industry Co., Ltd., Wistron Corporation, and Fabrinet to perform assembly."
)
NVIDIA = FilerContext("1045810", "NVIDIA Corporation", ("NVIDIA CORP",))
INTEL = FilerContext("50863", "Intel Corporation", ("INTEL CORP",))


def passage(text: str, cik: str = "1045810") -> Passage:
    return Passage(passage_id(cik, text), cik, text, 0, 0)


# --- text -------------------------------------------------------------------------------

def test_html_to_text_drops_hidden_xbrl_header_and_markup() -> None:
    raw = ("<html><ix:header><ix:hidden>dei:EntityName NVIDIA</ix:hidden></ix:header>"
           "<div>We&#160;purchase&nbsp;memory</div><script>x=1</script><p>from SK Hynix</p>")
    assert html_to_text(raw) == "We purchase memory from SK Hynix"


def test_sentences_do_not_split_inside_company_names() -> None:
    text = ("For example, ASML Holding N.V. (ASML) is currently the sole supplier of EUV tools. "
            "We buy from Hon Hai Precision Industry Co. Ltd. and others. Next sentence.")
    assert split_sentences(text) == [
        "For example, ASML Holding N.V. (ASML) is currently the sole supplier of EUV tools.",
        "We buy from Hon Hai Precision Industry Co. Ltd. and others.",
        "Next sentence.",
    ]


def test_bullets_start_new_sentences() -> None:
    text = "Our competitors include: • suppliers of GPUs such as AMD; • cloud companies"
    assert split_sentences(text) == ["Our competitors include:", "suppliers of GPUs such as AMD;",
                                     "cloud companies"]


def test_org_name_suffix_must_end_at_a_word_boundary() -> None:
    # Regression: "Inc" matched inside "Income"/"Incentive" and "SE" inside "SECURITIES".
    text = ("Consolidated Statements of Income. PURCHASES OF EQUITY SECURITIES BY THE ISSUER. "
            "Equity Incentive Plan.")
    assert org_names(text) == []
    assert org_names("We rely on Wistron Corporation.") == ["Wistron Corporation"]


def test_lists_of_names_are_split_at_corporate_suffixes() -> None:
    # Regression: the greedy match "SK Hynix Inc. and Kioxia Corporation" kept only Kioxia.
    assert org_names("We purchase memory from SK Hynix Inc. and Kioxia Corporation.") == [
        "SK Hynix Inc.", "Kioxia Corporation"]
    assert org_names("such as Hon Hai Precision Industry Co., Ltd., Wistron Corporation, "
                     "and Fabrinet") == ["Hon Hai Precision Industry Co., Ltd.",
                                         "Wistron Corporation"]
    # ...but a suffix followed by another suffix is one name.
    assert org_names("We use Taiwan Semiconductor Manufacturing Company Limited.") == [
        "Taiwan Semiconductor Manufacturing Company Limited"]


def test_self_references_glued_to_headings_are_not_company_names() -> None:
    # Regression: "Accrued Liabilities The Company" was extracted as a supplier.
    assert org_names("Accrued Liabilities The Company has supplier financing programs.") == []
    assert org_names("The Company relies on suppliers.") == []


def test_learn_aliases_from_definitions_anywhere_in_the_text() -> None:
    # Regression: an \A anchor made `.match(text, pos)` fail everywhere except position 0.
    text = ("Intro text. " + NVIDIA_SUPPLY + " Currently, we rely on suppliers such as Panasonic "
            "and Contemporary Amperex Technology Co. Limited (CATL) for these cells. Vanguard "
            "International Semiconductor Corporation (“VIS”) makes wafers.")
    aliases = learn_aliases(text)
    assert aliases["TSMC"] == "Taiwan Semiconductor Manufacturing Company Limited"
    assert aliases["Samsung"] == "Samsung Electronics Co., Ltd."
    # The greedy name match is trimmed to the entity the acronym actually stands for.
    assert aliases["CATL"] == "Contemporary Amperex Technology Co. Limited"
    assert aliases["VIS"] == "Vanguard International Semiconductor Corporation"


def test_implausible_aliases_are_rejected() -> None:
    aliases = learn_aliases("Cancellation of Unilever PLC (PLC) shares. Acme Widgets Inc. (XYZ).")
    assert "PLC" not in aliases  # a legal form, not a name
    assert "XYZ" not in aliases  # not an acronym of the name


def test_aliases_learned_in_one_filing_apply_to_all() -> None:
    merged = merge_aliases([{"TSMC": "Taiwan Semiconductor Manufacturing Company Limited"}, {}])
    sentence = "We expect to continue to use TSMC and other third-party foundry suppliers."
    [p] = candidate_passages(INTEL.cik, sentence, merged.keys(), self_names=INTEL.all_names)
    assert "TSMC" in p.text


def test_filer_mentioning_itself_is_not_a_candidate() -> None:
    aliases = {"ASML": "ASML Holding N.V.", "Zeiss": "Carl Zeiss SMT GmbH"}
    asml = FilerContext("937966", "ASML Holding N.V.", ("ASML HOLDING NV",))
    usable = aliases_excluding_filer(aliases, asml.all_names)
    assert usable == ["Zeiss"]
    text = "ASML supplies lithography systems to chipmakers. ASML Holding N.V. sources parts."
    assert candidate_passages(asml.cik, text, usable, self_names=asml.all_names) == []
    assert is_self("Overview Broadcom Inc.", ["Broadcom Inc."])  # greedy heading match


def test_candidate_windows_include_preceding_context_and_merge() -> None:
    text = ("Our current competitors include: • one • two • "
            "suppliers of GPUs such as Advanced Micro Devices, Inc. and Intel Corporation; "
            "• three. Unrelated. Unrelated two. Unrelated three. Unrelated four. "
            "We purchase memory from SK Hynix Inc.")
    p1, p2 = candidate_passages("1045810", text, self_names=NVIDIA.all_names)
    assert p1.text.startswith("Our current competitors include:")
    assert "SK Hynix" in p2.text and "competitors" not in p2.text


# --- grounding and rule baseline --------------------------------------------------------

def test_grounding_requires_verbatim_evidence_containing_the_name() -> None:
    text = "We purchase memory from SK Hynix Inc., Micron Technology, Inc., and Samsung."
    assert is_grounded("We purchase memory from SK Hynix Inc., Micron Technology, Inc., "
                       "and Samsung", "Micron Technology, Inc.", text)
    assert not is_grounded("We purchase memory from Kioxia.", "Kioxia", text)  # hallucinated
    assert not is_grounded(text, "Kioxia", text)  # real sentence, wrong company
    curly = "TSMC\N{RIGHT SINGLE QUOTATION MARK}s capacity is limited."
    assert is_grounded("TSMC's capacity is limited.", "TSMC", curly)


def test_rule_extractor_finds_suppliers_and_canonicalizes_aliases() -> None:
    aliases = learn_aliases(NVIDIA_SUPPLY)
    mentions = RuleExtractor(aliases).extract(passage(NVIDIA_SUPPLY), NVIDIA)
    names = {m.supplier_name for m in mentions}
    assert {"Taiwan Semiconductor Manufacturing Company Limited", "Samsung Electronics Co., Ltd.",
            "SK Hynix Inc.", "Micron Technology, Inc.", "Hon Hai Precision Industry Co., Ltd.",
            "Wistron Corporation"} <= names
    assert "Samsung" not in names and "TSMC" not in names  # aliases map to the full name
    assert "Fabrinet" not in names  # no corporate suffix: the baseline's known blind spot
    assert all(m.method == suppliers.RULES and m.evidence in NVIDIA_SUPPLY for m in mentions)


def test_rule_extractor_skips_competition_bios_and_tables() -> None:
    text = ("We face intense competition from well-established competitors such as Samsung "
            "Electronics Co., Ltd. and United Microelectronics Corporation. Prior to joining "
            "Intel, he worked at Micron Technology, Inc., a semiconductor manufacturing company "
            "and supplier. " + "Acme Supply Chain Company LLC 100 % " * 40)
    assert RuleExtractor({}).extract(passage(text, INTEL.cik), INTEL) == []


# --- Claude extractor (fake client; no network) -----------------------------------------

class FakeMessages:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def fake_client(outcome) -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(outcome)))


def response(relations: list[Relation] | None, stop_reason: str = "end_turn"):
    parsed = PassageRelations(relations=relations) if relations is not None else None
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed, model="fake-model")


def api_error(cls, status: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


COMPETITORS = ("Our current competitors include: suppliers of GPUs such as Advanced Micro "
               "Devices, Inc., or AMD, and Intel Corporation, or Intel. ")
MIXED = COMPETITORS + NVIDIA_SUPPLY

RELATIONS = [
    Relation(name="TSMC", relationship="SUPPLIER", supplies="wafers",
             evidence="We utilize foundries, such as Taiwan Semiconductor Manufacturing Company "
                      "Limited, or TSMC, and Samsung Electronics Co., Ltd., or Samsung, to "
                      "produce our semiconductor wafers."),
    Relation(name="Fabrinet", relationship="SUPPLIER", supplies="assembly",
             evidence="We engage with independent subcontractors and contract manufacturers "
                      "such as Hon Hai Precision Industry Co., Ltd., Wistron Corporation, and "
                      "Fabrinet to perform assembly."),
    Relation(name="Advanced Micro Devices, Inc.", alias="AMD", relationship="COMPETITOR",
             evidence=COMPETITORS.strip()),
    Relation(name="Kioxia Corporation", relationship="SUPPLIER", supplies="memory",
             evidence="We purchase memory from Kioxia Corporation."),  # hallucinated
]


def test_claude_extractor_keeps_grounded_suppliers_only() -> None:
    client = fake_client(response(RELATIONS))
    extractor = ClaudeExtractor(learn_aliases(MIXED), model="fake-model", client=client)
    mentions = extractor.extract(passage(MIXED), NVIDIA)

    assert {m.supplier_name for m in mentions} == {
        "Taiwan Semiconductor Manufacturing Company Limited", "Fabrinet"}
    assert extractor.ungrounded == 1  # Kioxia never reaches the graph
    [call] = client.beta.messages.calls
    assert call["output_format"] is PassageRelations
    assert call["fallbacks"] == "default" and call["betas"] == [suppliers.FALLBACK_BETA]
    assert json.loads(call["messages"][0]["content"])["filer"] == "NVIDIA Corporation"


@pytest.mark.parametrize("outcome", [
    response(None),
    response([], stop_reason="refusal"),
    api_error(anthropic.RateLimitError, 429),
    api_error(anthropic.InternalServerError, 500),
])
def test_claude_extractor_failures_return_none(outcome) -> None:
    extractor = ClaudeExtractor({}, model="fake-model", client=fake_client(outcome))
    assert extractor.extract(passage(NVIDIA_SUPPLY), NVIDIA) is None


def test_claude_extractor_raises_on_auth_errors() -> None:
    extractor = ClaudeExtractor({}, model="fake-model",
                                client=fake_client(api_error(anthropic.AuthenticationError, 401)))
    with pytest.raises(anthropic.AuthenticationError):
        extractor.extract(passage(NVIDIA_SUPPLY), NVIDIA)


def test_cached_extractor_reuses_answers_and_retries_failures() -> None:
    con = warehouse.connect(":memory:")
    client = fake_client(response(RELATIONS))
    cached = CachedClaudeExtractor(
        ClaudeExtractor(learn_aliases(MIXED), model="fake-model", client=client), con)
    first = cached.extract(passage(MIXED), NVIDIA)
    second = cached.extract(passage(MIXED), NVIDIA)
    assert first == second and (cached.calls, cached.hits) == (1, 1)
    assert len(client.beta.messages.calls) == 1

    failing = CachedClaudeExtractor(
        ClaudeExtractor({}, model="fake-model", client=fake_client(response(None))), con)
    other = passage("We buy cells from Panasonic Holdings Corporation.")
    assert failing.extract(other, NVIDIA) is None
    assert failing.extract(other, NVIDIA) is None
    assert failing.calls == 2  # failures are not cached


# --- evaluation scoring -----------------------------------------------------------------

def mention(cik: str, name: str, written: str | None = None) -> Mention:
    return Mention(cik, "p", name, written or name, None, "evidence", suppliers.LLM)


def test_score_matches_aliases_and_ignores_ambiguous_rows() -> None:
    gold = [
        GoldRow("1", "Taiwan Semiconductor Manufacturing Company Limited", ("TSMC",), "supplier",
                ""),
        GoldRow("1", "Fabrinet", (), "supplier", ""),
        GoldRow("1", "Xintec, Inc.", (), "ambiguous", ""),
        GoldRow("2", "ASML Holding N.V.", ("ASML",), "supplier", ""),
    ]
    predicted = [
        mention("1", "TSMC"), mention("1", "TSMC"),  # duplicates count once
        mention("1", "Xintec Inc"),  # ambiguous: neither TP nor FP
        mention("1", "Intel Corporation"),  # FP
        mention("2", "Fabrinet"),  # right name, wrong filer -> FP
    ]
    result = score(predicted, gold)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 2)
    assert result["precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert {m["supplier"] for m in result["missed"]} == {"Fabrinet", "ASML Holding N.V."}


def test_committed_eval_set_loads_and_rule_baseline_is_stable() -> None:
    from supplygraph.extract.evaluate import evaluate, load_eval_set

    eval_dir = Path(__file__).resolve().parents[1] / "eval" / "extraction"
    aliases, items, gold = load_eval_set(eval_dir)
    assert len(items) > 50 and sum(g.label == "supplier" for g in gold) >= 25
    rules = evaluate(eval_dir, {"rules": RuleExtractor(aliases)})["systems"]["rules"]
    # Documented in docs/extraction.md; if this moves, update the docs with the new numbers.
    assert (rules["tp"], rules["fp"], rules["fn"]) == (19, 15, 8)


# --- LEI linking ------------------------------------------------------------------------

SAMSUNG = make_lei("TESTSG0000000SAMSG")
SAMSUNG_NL = make_lei("TESTSG0000000SAMNL")
ETF = make_lei("TESTSG00000000ETF0")
PANA_HD = make_lei("TESTSG0000000PANHD")
PANA_OP = make_lei("TESTSG0000000PANOP")


def record(lei: str, name: str, other: list[str] | None = None, category: str = "GENERAL",
           country: str = "KR") -> dict:
    r = make_record(lei, name, country, other_names=other)
    r["attributes"]["entity"]["category"] = category
    return r


class FakeSearch:
    def __init__(self, results: dict[str, list[dict]], parents: dict[str, str] | None = None):
        self.results, self.parents, self.queries = results, parents or {}, []

    def search_fulltext(self, query: str, page_size: int = 20) -> list[dict]:
        self.queries.append(query)
        return self.results.get(query, [])

    def ultimate_parent(self, lei: str) -> dict | None:
        parent = self.parents.get(lei)
        return {"id": parent} if parent else None


def test_resolve_name_requires_an_exact_normalized_name() -> None:
    api = FakeSearch({"samsung electronics": [
        record(SAMSUNG_NL, "SAMSUNG ELECTRONICS BENELUX B.V.", country="NL"),
        record(ETF, "Samsung Electronics Daily 2x ETF", category="FUND"),
        record(SAMSUNG, "삼성전자(주)", other=["SAMSUNG ELECTRONICS CO., LTD"]),
    ]})
    res, rec = link.resolve_name(api, "Samsung Electronics Co., Ltd.")
    assert (res.status, res.lei) == (link.RESOLVED, SAMSUNG)
    assert rec["id"] == SAMSUNG


def test_resolve_name_respects_legal_form_and_unreadable_scripts() -> None:
    gf_ky, gf_us = make_lei("TESTSG00000000GFKY"), make_lei("TESTSG00000000GFUS")
    catl_de = make_lei("TESTSG0000000CATDE")
    greek = make_lei("TESTSG0000000SAMGR")
    api = FakeSearch({
        # Regression: "U.S." lost its single letters, so both normalized to "globalfoundries".
        "globalfoundries": [record(gf_us, "GLOBALFOUNDRIES U.S. INC.", country="US"),
                            record(gf_ky, "GLOBALFOUNDRIES INC.", country="KY")],
        # Regression: a German subsidiary matched the Chinese parent's name minus legal form.
        "contemporary amperex technology": [
            record(catl_de, "Contemporary Amperex Technology AG", country="DE")],
        # Regression: the Greek words vanished, leaving "samsung electronics".
        "samsung electronics": [
            record(greek, "SAMSUNG ELECTRONICS ΕΛΛΑΣ A.E.", country="GR"),
            record(SAMSUNG, "삼성전자(주)", other=["SAMSUNG ELECTRONICS CO., LTD"])],
    })
    assert link.resolve_name(api, "GLOBALFOUNDRIES Inc.")[0].lei == gf_ky
    assert link.resolve_name(api, "Samsung Electronics Co., Ltd.")[0].lei == SAMSUNG
    catl = link.resolve_name(api, "Contemporary Amperex Technology Co. Limited")[0]
    assert (catl.status, catl.lei) == (link.LEGAL_FORM_MISMATCH, None)
    # A shorter form of the same name is not a different legal form (Qualcomm's 10-K says
    # "Taiwan Semiconductor Manufacturing Company", GLEIF has "... Company Limited").
    tsmc = make_lei("TESTSG000000000TSM")
    api.results["taiwan semiconductor manufacturing"] = [record(
        tsmc, "台灣積體電路製造股份有限公司",
        other=["Taiwan Semiconductor Manufacturing Company Limited"], country="TW")]
    assert link.resolve_name(api, "Taiwan Semiconductor Manufacturing Company")[0].lei == tsmc
    # Both queries are tried: the normalized name and the name as written.
    assert "GLOBALFOUNDRIES Inc." in api.queries


def test_resolve_name_ties_go_to_the_group_parent_or_stay_ambiguous() -> None:
    results = {"panasonic": [record(PANA_HD, "Panasonic Holdings Corporation", country="JP"),
                             record(PANA_OP, "Panasonic Corporation", country="JP")]}
    res, _ = link.resolve_name(FakeSearch(results, {PANA_OP: PANA_HD}), "Panasonic")
    assert (res.status, res.lei) == (link.RESOLVED, PANA_HD)
    res, _ = link.resolve_name(FakeSearch(results), "Panasonic")  # unrelated namesakes
    assert (res.status, res.lei) == (link.AMBIGUOUS, None)
    res, _ = link.resolve_name(FakeSearch({}), "Fabrinet")
    assert res.status == link.NOT_FOUND


def test_resolution_cache_avoids_repeat_lookups(tmp_path) -> None:
    api = FakeSearch({"wistron": [record(SAMSUNG, "Wistron Corporation", country="TW")]})
    cache = link.ResolutionCache(tmp_path / "res.json")
    link.resolve_names(api, ["Wistron Corporation", "Wistron Corporation"], cache)
    first_run = list(api.queries)
    again = link.ResolutionCache(tmp_path / "res.json")
    out = link.resolve_names(api, ["Wistron Corporation"], again)
    assert api.queries == first_run == ["wistron", "Wistron Corporation"]  # one lookup only
    assert out["Wistron Corporation"][0].lei == SAMSUNG


def test_resolution_cache_is_discarded_when_the_rules_change(tmp_path) -> None:
    path = tmp_path / "res.json"
    path.write_text(json.dumps({"version": 0, "names": {"wistron": {"resolution": {}}}}))
    assert link.ResolutionCache(path).get("Wistron Corporation") is None


# --- SEC ingestion ----------------------------------------------------------------------

SUBMISSIONS = {
    "name": "NVIDIA CORP",
    "filings": {"recent": {
        "form": ["10-Q", "10-K/A", "10-K", "10-K"],
        "accessionNumber": ["a-4", "a-3", "0001045810-26-000021", "0001045810-25-000023"],
        "filingDate": ["2026-05-28", "2026-04-01", "2026-02-25", "2025-02-26"],
        "reportDate": ["2026-04-26", "2026-01-25", "2026-01-25", "2025-01-26"],
        "primaryDocument": ["q.htm", "amend.htm", "nvda-20260125.htm", "nvda-20250126.htm"],
    }},
}


def test_user_agent_must_contain_a_contact_email() -> None:
    for bad in (None, "", "supplygraph/0.1", "SupplyGraph research"):
        with pytest.raises(sec.SecConfigError):
            sec.check_user_agent(bad)
    assert sec.check_user_agent("SupplyGraph you@example.com")


def test_latest_annual_skips_amendments_and_quarterlies() -> None:
    row = sec.latest_annual(SUBMISSIONS)
    assert (row["form"], row["accessionNumber"]) == ("10-K", "0001045810-26-000021")
    assert sec.latest_annual({"filings": {"recent": {"form": ["8-K"]}}}) is None


def test_read_filers_rejects_bad_rows(tmp_path) -> None:
    path = tmp_path / "filers.csv"
    path.write_text("lei,cik,name\nNOT-A-LEI,1045810,NVIDIA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid rows"):
        sec.read_filers(path)


class FakeSec:
    def __init__(self) -> None:
        self.downloads: list[str] = []

    def submissions(self, cik: str) -> dict:
        return SUBMISSIONS if cik == "1045810" else {"filings": {"recent": {}}}

    def document(self, url: str) -> bytes:
        self.downloads.append(url)
        return b"<html>We purchase memory from SK Hynix Inc.</html>"


def test_ingest_filings_writes_manifest_and_never_redownloads(tmp_path) -> None:
    filers = [sec.Filer("549300S4KLFTLO7GSQ80", "1045810", "NVIDIA Corporation"),
              sec.Filer("HWUPKR0MPOU8FGXBT394", "320193", "Apple Inc.")]  # no annual report
    api = FakeSec()
    refs = sec.ingest_filings(api, filers, tmp_path)
    assert [r.accession for r in refs] == ["0001045810-26-000021"]
    assert refs[0].url == ("https://www.sec.gov/Archives/edgar/data/1045810/"
                           "000104581026000021/nvda-20260125.htm")
    assert (tmp_path / refs[0].path).read_bytes().startswith(b"<html>")
    assert sec.read_manifest(tmp_path) == refs
    sec.ingest_filings(api, filers, tmp_path)
    assert len(api.downloads) == 1  # filings are immutable


def test_sec_client_turns_403_into_a_config_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(403, request=request))
    client = sec.SecClient("SupplyGraph you@example.com", min_interval=0,
                           client=httpx.Client(transport=transport))
    with pytest.raises(sec.SecConfigError, match="403"):
        client.submissions("1045810")
