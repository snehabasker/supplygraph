from __future__ import annotations

import json

import pytest

from conftest import APPLE, NVIDIA, ROSNEFT, make_record
from supplygraph.ingest import gleif


@pytest.mark.parametrize("lei", [APPLE, NVIDIA, ROSNEFT])
def test_real_leis_pass_checksum(lei: str) -> None:
    assert gleif.is_valid_lei(lei)


@pytest.mark.parametrize(
    "lei",
    [
        APPLE[:-1] + ("5" if APPLE[-1] != "5" else "6"),  # wrong check digit
        APPLE.lower(),  # lowercase not allowed
        APPLE[:-1],  # too short
        "",
        None,
    ],
)
def test_invalid_leis_rejected(lei) -> None:
    assert not gleif.is_valid_lei(lei)


def test_parse_record_flattens_fields() -> None:
    row = gleif.parse_record(make_record(APPLE, "Apple Inc."))
    assert row["lei"] == APPLE
    assert row["legal_name"] == "Apple Inc."
    assert row["legal_country"] == "US"
    assert row["registration_status"] == "ISSUED"


def test_parse_record_tolerates_missing_and_null_sections() -> None:
    row = gleif.parse_record({"id": APPLE, "attributes": {"entity": {"legalAddress": None}}})
    assert row["lei"] == APPLE
    assert row["legal_name"] is None
    assert row["legal_country"] is None


class FakeApi:
    """In-memory GLEIF: NVIDIA's direct parent is Apple; Apple has child Rosneft
    (fictional, just to exercise every branch)."""

    def __init__(self, records: dict[str, dict]) -> None:
        self.records = records
        self.parents = {NVIDIA: APPLE}
        self.ultimate = {NVIDIA: APPLE}  # same as direct -> must not duplicate
        self.children = {APPLE: [ROSNEFT]}

    def get_record(self, lei):
        return self.records.get(lei)

    def direct_parent(self, lei):
        p = self.parents.get(lei)
        return self.records[p] if p else None

    def ultimate_parent(self, lei):
        p = self.ultimate.get(lei)
        return self.records[p] if p else None

    def direct_children(self, lei, limit):
        return [self.records[c] for c in self.children.get(lei, [])][:limit]


def test_crawl_collects_records_and_relationships(records) -> None:
    result = gleif.crawl(FakeApi(records), [NVIDIA, APPLE])
    assert set(result.records) == {APPLE, NVIDIA, ROSNEFT}
    assert result.relationships == {
        (NVIDIA, APPLE, gleif.DIRECT),
        (ROSNEFT, APPLE, gleif.DIRECT),
    }
    assert result.missing_seeds == []


def test_crawl_reports_missing_seeds_and_skips_self_loops(records) -> None:
    api = FakeApi(records)
    api.parents = {APPLE: APPLE}  # malformed upstream data
    result = gleif.crawl(api, [APPLE, "5493000000000000000X"], children_limit=0)
    assert result.relationships == set()
    assert result.missing_seeds == ["5493000000000000000X"]


def test_read_seed_leis_validates_and_dedupes(tmp_path) -> None:
    seeds = tmp_path / "seeds.csv"
    seeds.write_text(f"lei,name\n{APPLE},Apple\n{APPLE},dup\n{NVIDIA},Nvidia\n", encoding="utf-8")
    assert gleif.read_seed_leis(seeds) == [APPLE, NVIDIA]

    seeds.write_text("lei,name\nNOT-A-LEI,bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NOT-A-LEI"):
        gleif.read_seed_leis(seeds)


def test_write_crawl_produces_jsonl(tmp_path, records) -> None:
    result = gleif.crawl(FakeApi(records), [NVIDIA])
    records_path, rels_path = gleif.write_crawl(result, tmp_path)
    lines = records_path.read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["id"] for line in lines} == {APPLE, NVIDIA}
    rel = json.loads(rels_path.read_text(encoding="utf-8").splitlines()[0])
    assert rel == {"child_lei": NVIDIA, "parent_lei": APPLE, "relationship_type": "DIRECT"}
