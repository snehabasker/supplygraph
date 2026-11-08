from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from supplygraph.ingest import gleif

# Real, check-digit-valid LEIs (Apple, NVIDIA, Rosneft) so fixtures pass the same
# validation as production data.
APPLE = "HWUPKR0MPOU8FGXBT394"
NVIDIA = "549300S4KLFTLO7GSQ80"
ROSNEFT = "253400JT3MQWNDKMJE44"

SANCTIONS_HEADER = [
    "id", "schema", "name", "aliases", "birth_date", "countries", "addresses", "identifiers",
    "sanctions", "phones", "emails", "program_ids", "dataset", "first_seen", "last_seen",
    "last_change",
]


def make_lei(prefix18: str) -> str:
    """Build a check-digit-valid LEI from an 18-char prefix (ISO 7064 MOD 97-10)."""
    assert len(prefix18) == 18
    digits = "".join(str(int(ch, 36)) for ch in prefix18 + "00")
    return f"{prefix18}{98 - int(digits) % 97:02d}"


def make_record(
    lei: str,
    name: str,
    country: str = "US",
    other_names: list[str] | None = None,
    registered_as: str | None = None,
) -> dict:
    return {
        "type": "lei-records",
        "id": lei,
        "attributes": {
            "lei": lei,
            "entity": {
                "legalName": {"name": name, "language": "en"},
                "otherNames": [{"name": n, "language": "en",
                                "type": "ALTERNATIVE_LANGUAGE_LEGAL_NAME"}
                               for n in other_names or []],
                "registeredAs": registered_as,
                "legalAddress": {"city": "Somewhere", "country": country},
                "headquartersAddress": {"city": "Somewhere", "country": country},
                "jurisdiction": country,
                "category": "GENERAL",
                "legalForm": {"id": "XXXX", "other": None},
                "status": "ACTIVE",
            },
            "registration": {"status": "ISSUED", "lastUpdateDate": "2026-01-01T00:00:00Z"},
        },
    }


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def write_sanctions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SANCTIONS_HEADER, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SANCTIONS_HEADER})


def write_raw_data(
    data_dir: Path,
    records: dict[str, dict],
    relationships: set[tuple[str, str, str]],
    sanctions: list[dict],
) -> None:
    """Lay out raw files exactly where the pipeline expects them under `data_dir`."""
    gleif.write_crawl(gleif.CrawlResult(records=records, relationships=relationships),
                      data_dir / "raw" / "gleif")
    write_sanctions(data_dir / "raw" / "opensanctions" / "targets.simple.csv", sanctions)


def default_sanctions() -> list[dict]:
    return [
        # LEI hidden among other identifiers -> must match
        {"id": "NK-ros", "schema": "Organization", "name": "PJSC Rosneft",
         "identifiers": f"7706107510;{ROSNEFT};1027700043502", "last_seen": now_iso()},
        # LEI as a *substring* of a longer token -> must NOT match
        {"id": "NK-sub", "schema": "Company", "name": "Lookalike",
         "identifiers": f"X{APPLE}", "last_seen": now_iso()},
        {"id": "NK-person", "schema": "Person", "name": "Someone", "last_seen": now_iso()},
    ]


@pytest.fixture
def records() -> dict[str, dict]:
    return {
        APPLE: make_record(APPLE, "Apple Inc."),
        NVIDIA: make_record(NVIDIA, "NVIDIA CORPORATION"),
        ROSNEFT: make_record(ROSNEFT, "Rosneft Oil Company", country="RU"),
    }


@pytest.fixture
def raw_data_dir(tmp_path, records) -> Path:
    """A data dir with clean raw inputs: 3 companies, 1 ownership edge, 3 sanctions rows."""
    write_raw_data(tmp_path, records, {(NVIDIA, APPLE, gleif.DIRECT)}, default_sanctions())
    return tmp_path
