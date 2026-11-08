"""GDELT Global Knowledge Graph (GKG 2.0) ingestion: news articles tagged with disruption themes.

GDELT publishes a GKG file every 15 minutes (~1.7 MB zipped, a few thousand articles), with
named organizations and coded themes already extracted. We use these raw files rather than
the DOC search API, which rate-limits aggressively (one request per 5 s, and 429s on shared
IPs). Only articles that carry at least one disruption theme *and* name an organization are
kept, so a day of input (~160 MB) becomes a few MB of bronze data.

Codebook: http://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
import re
import zipfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import httpx

from supplygraph.http import get_with_retry

log = logging.getLogger(__name__)

GKG_URL = "https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
GKG_COLUMNS = 27
COL_ID, COL_DATE, COL_DOMAIN, COL_URL = 0, 1, 3, 4
COL_THEMES, COL_LOCATIONS, COL_ORGS, COL_TONE, COL_EXTRAS = 7, 9, 13, 15, 26
# "Enhanced" columns carry character offsets into the article: "THEME,1234;..." / "Name,567;..."
COL_V2_THEMES, COL_V2_ORGS = 8, 14

# Themes are tagged per *article*, so a market-wrap story with one storm metaphor is a
# "natural disaster" that "mentions NVIDIA". On real files that made ~90% of company links
# irrelevant. An organization therefore only counts if it is named within this many
# characters (roughly a sentence or two) of a disruption theme.
PROXIMITY_CHARS = 400

# Disruption event types, from GKG theme codes. Chosen from theme frequencies on real files:
# MANMADE_DISASTER_IMPLIED is deliberately absent — it tags ~19% of *all* articles.
EXACT_THEMES = {
    "STRIKE": "LABOR_STRIKE",
    "SHORTAGE": "SHORTAGE",
    "SANCTIONS": "SANCTIONS",
    "CYBER_ATTACK": "CYBER_ATTACK",
    "DISASTER_FIRE": "FIRE",
    "MANMADE_DISASTER": "INDUSTRIAL_ACCIDENT",
    "POWER_OUTAGE": "POWER_OUTAGE",
    "MANMADE_DISASTER_POWER_OUTAGES": "POWER_OUTAGE",
    "BLOCKADE": "BLOCKADE",
    "ECON_BANKRUPTCY": "BANKRUPTCY",
}
PREFIX_THEMES = {"NATURAL_DISASTER": "NATURAL_DISASTER"}
EVENT_TYPES = tuple(sorted({*EXACT_THEMES.values(), *PREFIX_THEMES.values()}))

_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


def event_types(themes: Iterable[str]) -> list[str]:
    found = set()
    for theme in themes:
        if theme in EXACT_THEMES:
            found.add(EXACT_THEMES[theme])
            continue
        for prefix, event_type in PREFIX_THEMES.items():
            if theme == prefix or theme.startswith(prefix + "_"):
                found.add(event_type)
    return sorted(found)


def stamps(end: datetime, hours: float) -> list[str]:
    """15-minute GKG file stamps covering (end - hours, end], newest first."""
    end = end.astimezone(UTC).replace(second=0, microsecond=0)
    end -= timedelta(minutes=end.minute % 15)
    count = max(1, int(hours * 4))
    return [(end - timedelta(minutes=15 * i)).strftime("%Y%m%d%H%M%S") for i in range(count)]


def _published_at(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC).isoformat()
    except ValueError:
        return None


def _locations(value: str, limit: int = 5) -> list[str]:
    # V1Locations: "type#full name#country code#adm1#lat#long#feature id;..."
    names = [part.split("#")[1] for part in value.split(";") if part.count("#") >= 2]
    return list(dict.fromkeys(n for n in names if n))[:limit]


def _offsets(value: str) -> list[tuple[str, int]]:
    """"Name,123;Other, Inc.,456" -> [("Name", 123), ("Other, Inc.", 456)]."""
    out = []
    for item in value.split(";"):
        name, _, offset = item.rpartition(",")
        if name and offset.isdigit():
            out.append((name.strip(), int(offset)))
    return out


def near_disruption(
    themes: list[tuple[str, int]], orgs: list[tuple[str, int]], window: int = PROXIMITY_CHARS
) -> tuple[list[str], list[str]]:
    """(organizations named within `window` chars of a disruption theme, those themes'
    event types)."""
    hits = [(t, off) for t, off in themes if event_types([t])]
    kept, types = {}, set()
    for org, org_off in orgs:
        close = [t for t, off in hits if abs(off - org_off) <= window]
        if close:
            kept.setdefault(org.lower(), None)
            types.update(event_types(close))
    return list(kept), sorted(types)


def parse_row(row: list[str]) -> dict | None:
    """A GKG row -> event dict, or None unless an organization is named close to a
    disruption theme."""
    if len(row) != GKG_COLUMNS:
        return None
    orgs, types = near_disruption(_offsets(row[COL_V2_THEMES]), _offsets(row[COL_V2_ORGS]))
    if not types or not orgs:
        return None
    title = _TITLE.search(row[COL_EXTRAS])
    tone = row[COL_TONE].split(",")[0]
    try:
        tone_value = float(tone) if tone else None
    except ValueError:
        tone_value = None
    return {
        "id": row[COL_ID],
        "published_at": _published_at(row[COL_DATE]),
        "domain": row[COL_DOMAIN] or None,
        "url": row[COL_URL] or None,
        "title": html.unescape(title.group(1)).strip() if title else None,
        "event_types": types,
        "organizations": orgs,
        "tone": tone_value,
        "locations": _locations(row[COL_LOCATIONS]),
    }


def parse_gkg_zip(payload: bytes) -> Iterator[dict]:
    csv.field_size_limit(1 << 30)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            text = zf.read(member).decode("utf-8", errors="replace")
            # GKG is tab-separated with no quoting; quotes inside titles are literal.
            for row in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
                event = parse_row(row)
                if event is not None:
                    yield event


class GkgSource(Protocol):
    def fetch(self, stamp: str) -> bytes | None: ...


class GdeltClient:
    def __init__(self, user_agent: str, *, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(headers={"User-Agent": user_agent},
                                              timeout=60.0, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GdeltClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(self, stamp: str) -> bytes | None:
        """The zipped GKG file, or None if GDELT skipped that interval (it occasionally does)."""
        resp = get_with_retry(self._client, GKG_URL.format(stamp=stamp))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content


def ingest_news(source: GkgSource, end: datetime, hours: float, out_dir: Path) -> dict[str, int]:
    """Write `events.jsonl` (full refresh for the window) and return counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    files = missing = 0
    tmp = out_dir / "events.jsonl.part"
    with tmp.open("w", encoding="utf-8") as fh:
        for stamp in stamps(end, hours):
            payload = source.fetch(stamp)
            if payload is None:
                missing += 1
                continue
            files += 1
            for event in parse_gkg_zip(payload):
                if event["id"] in seen:
                    continue
                seen.add(event["id"])
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    tmp.replace(out_dir / "events.jsonl")
    log.info("GDELT: %d files (%d missing), %d disruption events", files, missing, len(seen))
    return {"files": files, "missing_files": missing, "events": len(seen)}
