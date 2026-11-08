"""Supply-chain warehouse tables, their quality checks, and the end-to-end extraction step."""

from __future__ import annotations

import json

import pytest

from conftest import (
    APPLE,
    NVIDIA,
    filing_row,
    make_lei,
    make_record,
    mention_row,
    news_event,
    write_jsonl,
)
from supplygraph import quality, warehouse
from supplygraph.extract import supply_chain
from supplygraph.extract.suppliers import RuleExtractor
from supplygraph.ingest import sec

TSMC = make_lei("TESTSG000000000TSM")
HYNIX = make_lei("TESTSG00000000HYNX")


@pytest.fixture
def chain_dir(raw_data_dir):
    """raw_data_dir + a supplier crawl, an NVIDIA filing, mentions and news."""
    raw = raw_data_dir / "raw"
    write_jsonl(raw / "gleif" / "suppliers" / "records.jsonl", [
        make_record(TSMC, "Taiwan Semiconductor Manufacturing Company Limited", "TW"),
        make_record(HYNIX, "SK hynix Inc.", "KR"),
        make_record(NVIDIA, "NVIDIA CORPORATION"),  # already in the seed crawl: deduplicated
    ])
    write_jsonl(raw / "gleif" / "suppliers" / "relationships.jsonl", [])
    (raw / "gleif" / "seeds.json").write_text(json.dumps([APPLE]))
    write_jsonl(raw / "sec" / "filings.jsonl", [filing_row(NVIDIA, "1045810", "NVIDIA Corp")])
    write_jsonl(raw / "sec" / "supplier_mentions.jsonl", [
        mention_row(NVIDIA, "1045810", "Taiwan Semiconductor Manufacturing Company Limited",
                    TSMC, "LLM", passage="p1"),
        mention_row(NVIDIA, "1045810", "Taiwan Semiconductor Manufacturing Company Limited",
                    TSMC, "LLM", evidence="TSMC makes our wafers.", passage="p2"),
        mention_row(NVIDIA, "1045810", "SK hynix Inc.", HYNIX, "RULES"),
        mention_row(NVIDIA, "1045810", "NVIDIA Corporation", NVIDIA, "RULES"),  # self loop
        mention_row(NVIDIA, "1045810", "Fabrinet", None, "LLM"),  # unresolved
    ])
    (raw / "sec" / "aliases.json").write_text(json.dumps(
        {"TSMC": "Taiwan Semiconductor Manufacturing Company Limited"}))
    write_jsonl(raw / "gdelt" / "events.jsonl", [
        news_event("E1", ["tsmc"], ["NATURAL_DISASTER"]),
        news_event("E2", ["unrelated org"]),
    ])
    return raw_data_dir


def build(data_dir):
    con = warehouse.connect(":memory:")
    raw = data_dir / "raw"
    counts = warehouse.build_all(con, raw / "gleif", raw / "opensanctions" / "targets.simple.csv",
                                 raw / "sec", raw / "gdelt")
    return con, counts


def test_supply_links_follow_the_extraction_method(chain_dir) -> None:
    con, counts = build(chain_dir)
    assert counts["companies"] == 5  # 3 seeds-crawl + 2 suppliers, NVIDIA not duplicated
    rows = con.execute("""SELECT supplier_lei, link_type, mentions, evidence FROM supply_links
                          ORDER BY link_type""").fetchall()
    assert rows == [
        (HYNIX, "POSSIBLE_SUPPLIER", 1, "We buy wafers from SK hynix Inc."),
        # Two mentions aggregate into one link; the shortest evidence sentence is kept.
        (TSMC, "SUPPLIES", 2, "TSMC makes our wafers."),
    ]


def test_watchlist_and_news_links(chain_dir) -> None:
    con, counts = build(chain_dir)
    reasons = dict(con.execute("SELECT lei, reasons FROM watchlist").fetchall())
    # SK hynix is only a POSSIBLE_SUPPLIER lead: leads don't expand monitoring.
    assert reasons == {APPLE: "seed", NVIDIA: "customer;sec_filer", TSMC: "supplier"}
    assert con.execute("SELECT lei, event_id, matched_name FROM company_news").fetchall() == [
        (TSMC, "E1", "tsmc")]
    assert counts["news_events"] == 2


def test_supply_chain_checks_pass_on_clean_data(chain_dir) -> None:
    con, _ = build(chain_dir)
    results = {r.name: r for r in quality.run_checks(con)}
    for name in ("supply_links_no_orphans", "supply_links_no_self_loops",
                 "supply_links_type_follows_method", "supply_links_have_evidence",
                 "mentions_method_known", "news_events_typed", "company_news_references_valid"):
        assert results[name].status == "pass", name
    assert not quality.has_blocking_failures(list(results.values()))


@pytest.mark.parametrize(("sql", "check"), [
    ("UPDATE supply_links SET link_type = 'SUPPLIES' WHERE extraction_method = 'RULES'",
     "supply_links_type_follows_method"),
    ("UPDATE supply_links SET evidence = ' '", "supply_links_have_evidence"),
    ("UPDATE supplier_mentions SET method = 'GUESS'", "mentions_method_known"),
    ("UPDATE news_events SET event_types = []", "news_events_typed"),
    ("INSERT INTO company_news VALUES ('NOT-WATCHED', 'E1', 'x')",
     "company_news_references_valid"),
])
def test_supply_chain_checks_catch_corruption(chain_dir, sql, check) -> None:
    con, _ = build(chain_dir)
    con.execute(sql)
    results = {r.name: r for r in quality.run_checks(con)}
    assert results[check].status == "fail"
    assert quality.has_blocking_failures(list(results.values()))


def test_build_without_sec_or_news_inputs_still_works(raw_data_dir) -> None:
    con, counts = build(raw_data_dir)
    assert counts["filings"] == counts["supply_links"] == counts["company_news"] == 0
    assert not quality.has_blocking_failures(quality.run_checks(con))


# --- end-to-end extraction step (fake GLEIF, real parsing) ------------------------------

FILING_HTML = """<html><body><div>We utilize foundries, such as Taiwan Semiconductor
Manufacturing Company Limited, or TSMC, to produce our semiconductor wafers.</div>
<div>We purchase memory from SK Hynix Inc. and Kioxia Corporation.</div></body></html>"""


class FakeGleif:
    def __init__(self) -> None:
        self.records = {
            "taiwan semiconductor manufacturing": make_record(
                TSMC, "台灣積體電路製造股份有限公司", "TW",
                other_names=["Taiwan Semiconductor Manufacturing Company Limited"]),
            "sk hynix": make_record(HYNIX, "에스케이하이닉스 주식회사", "KR",
                                    other_names=["SK HYNIX INC."]),
        }

    def search_fulltext(self, query, page_size=20):
        return [self.records[query]] if query in self.records else []

    def get_record(self, lei):
        return next((r for r in self.records.values() if r["id"] == lei), None)

    def direct_parent(self, lei):
        return None

    def ultimate_parent(self, lei):
        return None

    def direct_children(self, lei, limit):
        return []


def test_extraction_step_writes_resolved_mentions_and_supplier_records(tmp_path) -> None:
    sec_dir, gleif_dir = tmp_path / "sec", tmp_path / "gleif" / "suppliers"
    ref = sec.FilingRef(NVIDIA, "1045810", "NVIDIA Corporation", "NVIDIA CORP", "10-K",
                        "0001045810-26-000021", "2026-02-25", "2026-01-25", "https://x",
                        "1045810/doc.htm")
    (sec_dir / "1045810").mkdir(parents=True)
    (sec_dir / ref.path).write_text(FILING_HTML, encoding="utf-8")

    texts = supply_chain.load_texts(sec_dir, [ref])
    aliases, _ = supply_chain.build_passages([ref], texts)
    counts = supply_chain.run(sec_dir, gleif_dir, [ref], RuleExtractor(aliases), FakeGleif())

    assert counts["resolved"] == 2 and counts["unresolved"] == 1  # Kioxia: not in GLEIF
    rows = [json.loads(line) for line in
            (sec_dir / "supplier_mentions.jsonl").read_text(encoding="utf-8").splitlines()]
    by_name = {r["supplier_name"]: r for r in rows}
    assert by_name["Taiwan Semiconductor Manufacturing Company Limited"]["supplier_lei"] == TSMC
    assert by_name["Kioxia Corporation"]["resolution_status"] == "not_found"
    supplier_ids = {json.loads(line)["id"] for line in
                    (gleif_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()}
    assert supplier_ids == {TSMC, HYNIX}
    assert json.loads((sec_dir / "aliases.json").read_text(encoding="utf-8"))["TSMC"]
