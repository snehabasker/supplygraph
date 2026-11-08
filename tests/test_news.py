"""GDELT GKG parsing/ingestion and matching news to watchlist companies."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime

from conftest import make_lei, make_record, write_jsonl
from supplygraph import warehouse
from supplygraph.ingest import gdelt
from supplygraph.news import disambiguate, match_events


def gkg_row(record_id: str, themes: str, orgs: str, title: str = "Fab fire") -> list[str]:
    """`themes` / `orgs` use the enhanced "NAME,offset;..." format."""
    row = [""] * gdelt.GKG_COLUMNS
    row[gdelt.COL_ID] = record_id
    row[gdelt.COL_DATE] = "20261108011500"
    row[gdelt.COL_DOMAIN] = "example.com"
    row[gdelt.COL_URL] = f"https://example.com/{record_id}"
    row[gdelt.COL_THEMES] = ";".join(t.split(",")[0] for t in themes.split(";"))
    row[gdelt.COL_V2_THEMES] = themes
    row[gdelt.COL_LOCATIONS] = "4#Hsinchu, Taiwan#TW#TW00#24.8#120.9#-1;4#Hsinchu, Taiwan#TW#"
    row[gdelt.COL_ORGS] = ";".join(o.rsplit(",", 1)[0].lower() for o in orgs.split(";") if o)
    row[gdelt.COL_V2_ORGS] = orgs
    row[gdelt.COL_TONE] = "-4.2,1.0,5.2,6.2,20.1,0,300"
    row[gdelt.COL_EXTRAS] = f"<PAGE_TITLE>{title}</PAGE_TITLE>"
    return row


def gkg_zip(rows: list[list[str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.gkg.csv", "\n".join("\t".join(r) for r in rows))
    return buf.getvalue()


def test_event_types_use_an_explicit_allow_list() -> None:
    assert gdelt.event_types(["NATURAL_DISASTER_EARTHQUAKE", "STRIKE", "WB_135_TRANSPORT"]) == [
        "LABOR_STRIKE", "NATURAL_DISASTER"]
    # Tags ~19% of all articles: deliberately not a disruption signal.
    assert gdelt.event_types(["MANMADE_DISASTER_IMPLIED", "PROTEST"]) == []
    assert gdelt.event_types(["NATURAL_DISASTERS_FOO"]) == []  # prefix needs an underscore
    assert set(gdelt.event_types(gdelt.EXACT_THEMES)) <= set(gdelt.EVENT_TYPES)


def test_stamps_cover_the_window_in_15_minute_steps() -> None:
    end = datetime(2026, 11, 8, 2, 7, 33, tzinfo=UTC)
    assert gdelt.stamps(end, 1) == ["20261108020000", "20261108014500", "20261108013000",
                                    "20261108011500"]
    assert len(gdelt.stamps(end, 24)) == 96


def test_parse_row_extracts_fields_and_skips_irrelevant_rows() -> None:
    event = gdelt.parse_row(gkg_row(
        "A-1", "DISASTER_FIRE,120;TAX_FNCACT,10",
        "TSMC,100;Hsinchu Science Park,300;TSMC,900", title="Fab fire at TSMC &amp; partners"))
    assert event == {
        "id": "A-1", "published_at": "2026-11-08T01:15:00+00:00", "domain": "example.com",
        "url": "https://example.com/A-1", "title": "Fab fire at TSMC & partners",
        "event_types": ["FIRE"], "organizations": ["tsmc", "hsinchu science park"],
        "tone": -4.2, "locations": ["Hsinchu, Taiwan"],
    }
    assert gdelt.parse_row(gkg_row("A-2", "WB_135_TRANSPORT,5", "TSMC,10")) is None  # no theme
    assert gdelt.parse_row(gkg_row("A-3", "STRIKE,5", "")) is None  # no organization
    assert gdelt.parse_row(["too", "short"]) is None


def test_organizations_must_be_named_near_the_disruption() -> None:
    # Regression: article-level themes linked NVIDIA to a "natural disaster" in a market
    # wrap whose only disaster was a metaphor 5,000 characters away.
    row = gkg_row("M-1", "NATURAL_DISASTER_TSUNAMI,5200;SHORTAGE,80",
                  "Nvidia,60;Intel Corporation,4000;Acme, Inc.,5300")
    event = gdelt.parse_row(row)
    assert event["organizations"] == ["nvidia", "acme, inc."]  # comma inside a name survives
    assert event["event_types"] == ["NATURAL_DISASTER", "SHORTAGE"]
    far = gkg_row("M-2", "NATURAL_DISASTER,5200", "Nvidia,60")
    assert gdelt.parse_row(far) is None


def test_syndicated_copies_collapse_to_one_event() -> None:
    from supplygraph.warehouse import dedupe_syndicated

    a = {"id": "2", "published_at": "2026-11-08T02:00:00+00:00", "title": "Fab Fire  at TSMC",
         "organizations": ["tsmc"], "event_types": ["FIRE"]}
    b = {"id": "1", "published_at": "2026-11-08T01:00:00+00:00", "title": "fab fire at tsmc",
         "organizations": ["taiwan semiconductor"], "event_types": ["POWER_OUTAGE"]}
    c = {"id": "3", "published_at": None, "title": None, "organizations": ["x"],
         "event_types": ["FIRE"]}
    out = {e["id"]: e for e in dedupe_syndicated([a, b, c])}
    assert set(out) == {"1", "3"}  # earliest copy kept; untitled events never merge
    assert out["1"]["copies"] == 2
    assert out["1"]["organizations"] == ["taiwan semiconductor", "tsmc"]
    assert out["1"]["event_types"] == ["FIRE", "POWER_OUTAGE"]


class FakeGkg:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files, self.fetched = files, []

    def fetch(self, stamp: str) -> bytes | None:
        self.fetched.append(stamp)
        return self.files.get(stamp)  # missing interval -> None, like GDELT's 404


def test_ingest_news_dedupes_counts_missing_files_and_writes_jsonl(tmp_path) -> None:
    end = datetime(2026, 11, 8, 2, 0, tzinfo=UTC)
    newest, older = gdelt.stamps(end, 0.5)
    source = FakeGkg({
        newest: gkg_zip([gkg_row("A-1", "SHORTAGE,10", "Micron,20"),
                         gkg_row("A-2", "PROTEST,10", "X,20")]),
        older: gkg_zip([gkg_row("A-1", "SHORTAGE,10", "Micron,20")]),  # re-published record
    })
    counts = gdelt.ingest_news(source, end, 0.75, tmp_path)  # 3 stamps, 1 missing
    assert counts == {"files": 2, "missing_files": 1, "events": 1}
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["A-1"]


def test_gdelt_client_treats_404_as_a_missing_interval() -> None:
    import httpx

    transport = httpx.MockTransport(lambda r: httpx.Response(404, request=r))
    client = gdelt.GdeltClient("test", client=httpx.Client(transport=transport))
    assert client.fetch("20261108020000") is None


# --- matching --------------------------------------------------------------------------

TSMC = make_lei("TESTSG000000000TSM")
BASF_SE = make_lei("TESTSG00000000BASF")
BASF_US = make_lei("TESTSG0000000BASUS")
ACME_A = make_lei("TESTSG0000000ACMEA")
ACME_B = make_lei("TESTSG0000000ACMEB")
ATLAS = make_lei("TESTSG0000000ATLAS")


def news_con(tmp_path):
    records = {
        TSMC: make_record(TSMC, "台灣積體電路製造股份有限公司", "TW",
                          other_names=["Taiwan Semiconductor Manufacturing Company Limited"]),
        BASF_SE: make_record(BASF_SE, "BASF SE", "DE"),
        BASF_US: make_record(BASF_US, "BASF Corporation", "US"),
        ACME_A: make_record(ACME_A, "Acme Ltd", "GB"),
        ACME_B: make_record(ACME_B, "Acme Inc.", "US"),
        ATLAS: make_record(ATLAS, "Atlas GmbH", "DE"),  # crawled, but not on the watchlist
    }
    gleif_dir = tmp_path / "raw" / "gleif"
    write_jsonl(gleif_dir / "records.jsonl", list(records.values()))
    write_jsonl(gleif_dir / "relationships.jsonl", [
        {"child_lei": BASF_US, "parent_lei": BASF_SE, "relationship_type": "DIRECT"}])
    (gleif_dir / "seeds.json").write_text(json.dumps([TSMC, BASF_SE, BASF_US, ACME_A, ACME_B]))
    con = warehouse.connect(":memory:")
    warehouse.build_gleif(con, gleif_dir)
    warehouse.build_supply_chain(con, tmp_path / "raw" / "sec")  # empty: no filings
    warehouse.build_watchlist(con, gleif_dir)
    return con


def test_news_matches_watchlist_names_aliases_and_group_parents(tmp_path) -> None:
    con = news_con(tmp_path)
    events = [
        {"id": "E1", "organizations": ["tsmc", "taiwan semiconductor manufacturing company"]},
        {"id": "E2", "organizations": ["basf"]},  # BASF SE and BASF Corporation: parent wins
        {"id": "E3", "organizations": ["acme"]},  # two unrelated Acmes: ambiguous, skipped
        {"id": "E4", "organizations": ["atlas"]},  # not on the watchlist
    ]
    aliases = {"TSMC": "Taiwan Semiconductor Manufacturing Company Limited"}
    rows, stats = match_events(con, events, aliases)
    assert rows == sorted([[TSMC, "E1", "tsmc"], [BASF_SE, "E2", "basf"]])
    assert stats["ambiguous_hits"] == 1


def test_disambiguate_prefers_the_parent_only_when_unique() -> None:
    assert disambiguate({"A"}, {}) == "A"
    assert disambiguate({"P", "C"}, {"C": {"P"}}) == "P"
    assert disambiguate({"A", "B"}, {}) is None
