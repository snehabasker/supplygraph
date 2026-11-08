"""GLEIF ingestion: legal entities (LEI records) and their ownership relationships.

API docs: https://www.gleif.org/en/lei-data/gleif-api  (data licensed CC0)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from supplygraph.http import RateLimiter, get_with_retry

log = logging.getLogger(__name__)

LEI_PATTERN = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")

DIRECT = "DIRECT"
ULTIMATE = "ULTIMATE"


def is_valid_lei(lei: str) -> bool:
    """Validate format and ISO 17442 / ISO 7064 MOD 97-10 check digits."""
    if not isinstance(lei, str) or not LEI_PATTERN.match(lei):
        return False
    digits = "".join(str(int(ch, 36)) for ch in lei)
    return int(digits) % 97 == 1


def parse_record(raw: dict) -> dict:
    """Flatten a GLEIF `lei-records` resource into a warehouse row. Missing fields become None."""
    attrs = raw.get("attributes") or {}
    entity = attrs.get("entity") or {}
    registration = attrs.get("registration") or {}
    legal_address = entity.get("legalAddress") or {}
    hq_address = entity.get("headquartersAddress") or {}
    other_names = [
        n["name"]
        for key in ("otherNames", "transliteratedOtherNames")
        for n in (entity.get(key) or [])
        if isinstance(n, dict) and n.get("name")
    ]
    return {
        "lei": attrs.get("lei") or raw.get("id"),
        "legal_name": (entity.get("legalName") or {}).get("name"),
        # ';'-joined, matching the OpenSanctions convention. Used by entity resolution.
        "other_names": ";".join(dict.fromkeys(other_names)) or None,
        # Local business-register number (e.g. Russian OGRN, UK company number).
        "registered_as": entity.get("registeredAs"),
        "jurisdiction": entity.get("jurisdiction"),
        "legal_country": legal_address.get("country"),
        "legal_city": legal_address.get("city"),
        "hq_country": hq_address.get("country"),
        "hq_city": hq_address.get("city"),
        "category": entity.get("category"),
        "legal_form_id": (entity.get("legalForm") or {}).get("id"),
        "entity_status": entity.get("status"),
        "registration_status": registration.get("status"),
        "last_update": registration.get("lastUpdateDate"),
    }


_ENTITY_FIELDS = ("legalName", "otherNames", "transliteratedOtherNames", "jurisdiction",
                  "category", "legalForm", "status", "registeredAs")


def compact_record(raw: dict) -> dict:
    """Keep only what `parse_record` reads (for committed fixtures / eval sets)."""
    attrs = raw.get("attributes") or {}
    entity = attrs.get("entity") or {}
    registration = attrs.get("registration") or {}
    compact_entity = {k: entity[k] for k in _ENTITY_FIELDS if entity.get(k)}
    for key in ("legalAddress", "headquartersAddress"):
        address = entity.get(key) or {}
        compact_entity[key] = {k: address.get(k) for k in ("city", "country")}
    return {
        "id": raw.get("id"),
        "attributes": {
            "lei": attrs.get("lei"),
            "entity": compact_entity,
            "registration": {k: registration.get(k) for k in ("status", "lastUpdateDate")},
        },
    }


class GleifApi(Protocol):
    def get_record(self, lei: str) -> dict | None: ...
    def direct_parent(self, lei: str) -> dict | None: ...
    def ultimate_parent(self, lei: str) -> dict | None: ...
    def direct_children(self, lei: str, limit: int) -> list[dict]: ...


class GleifClient:
    """Thin, rate-limited client for the endpoints SupplyGraph needs."""

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        *,
        min_interval: float = 1.1,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "application/vnd.api+json"},
            timeout=30.0,
        )
        self._limiter = RateLimiter(min_interval)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GleifClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        self._limiter.wait()
        resp = get_with_retry(self._client, f"{self.base_url}{path}", params=params)
        if resp.status_code == 404:  # GLEIF uses 404 for "no such relationship"
            return None
        resp.raise_for_status()
        return resp.json()

    def get_record(self, lei: str) -> dict | None:
        body = self._get(f"/lei-records/{lei}")
        return body["data"] if body else None

    def direct_parent(self, lei: str) -> dict | None:
        body = self._get(f"/lei-records/{lei}/direct-parent")
        return body["data"] if body else None

    def ultimate_parent(self, lei: str) -> dict | None:
        body = self._get(f"/lei-records/{lei}/ultimate-parent")
        return body["data"] if body else None

    def direct_children(self, lei: str, limit: int) -> list[dict]:
        if limit <= 0:
            return []
        body = self._get(f"/lei-records/{lei}/direct-children", {"page[size]": min(limit, 200)})
        return body["data"] if body else []

    def list_by_country(self, country: str, page: int, page_size: int = 100) -> list[dict]:
        """One page of entities whose legal address is in `country` (ISO alpha-2)."""
        body = self._get("/lei-records", {
            "filter[entity.legalAddress.country]": country,
            "page[size]": page_size,
            "page[number]": page,
        })
        return body["data"] if body else []

    def search_fulltext(self, query: str, page_size: int = 20) -> list[dict]:
        """Name search across legal and other names (incl. English names of entities whose
        legal name is in a native script, e.g. 삼성전자(주) / SAMSUNG ELECTRONICS CO., LTD).
        Results are loose: callers must verify the name themselves."""
        body = self._get("/lei-records", {"filter[fulltext]": query, "page[size]": page_size})
        return body["data"] if body else []

    def get_records(self, leis: list[str], chunk_size: int = 100) -> list[dict]:
        """Batch lookup: one request per `chunk_size` LEIs instead of one per LEI."""
        records: list[dict] = []
        for i in range(0, len(leis), chunk_size):
            chunk = leis[i : i + chunk_size]
            body = self._get("/lei-records",
                             {"filter[lei]": ",".join(chunk), "page[size]": len(chunk)})
            records.extend(body["data"] if body else [])
        return records


@dataclass
class CrawlResult:
    records: dict[str, dict] = field(default_factory=dict)  # lei -> raw GLEIF resource
    relationships: set[tuple[str, str, str]] = field(default_factory=set)  # (child, parent, type)
    missing_seeds: list[str] = field(default_factory=list)


def crawl(api: GleifApi, seed_leis: Iterable[str], children_limit: int = 25) -> CrawlResult:
    """Fetch each seed entity plus its direct/ultimate parents and (some) direct children."""
    result = CrawlResult()

    def add_rel(child: str, parent: str, rel_type: str) -> None:
        if child != parent:
            result.relationships.add((child, parent, rel_type))

    for lei in seed_leis:
        record = api.get_record(lei)
        if record is None:
            log.warning("Seed LEI %s not found in GLEIF", lei)
            result.missing_seeds.append(lei)
            continue
        result.records[lei] = record

        direct = api.direct_parent(lei)
        if direct is not None:
            result.records[direct["id"]] = direct
            add_rel(lei, direct["id"], DIRECT)

        ultimate = api.ultimate_parent(lei)
        if ultimate is not None and (direct is None or ultimate["id"] != direct["id"]):
            result.records[ultimate["id"]] = ultimate
            add_rel(lei, ultimate["id"], ULTIMATE)

        for child in api.direct_children(lei, children_limit):
            result.records[child["id"]] = child
            add_rel(child["id"], lei, DIRECT)

        log.info("GLEIF %s: %d records, %d relationships so far",
                 lei, len(result.records), len(result.relationships))
    return result


def read_seed_leis(path: Path) -> list[str]:
    """Read the `lei` column of a seeds CSV, rejecting malformed identifiers early."""
    import csv

    with path.open(encoding="utf-8", newline="") as fh:
        leis = [row["lei"].strip() for row in csv.DictReader(fh) if row.get("lei", "").strip()]
    bad = [lei for lei in leis if not is_valid_lei(lei)]
    if bad:
        raise ValueError(f"Invalid LEIs in {path}: {bad}")
    return list(dict.fromkeys(leis))  # de-duplicate, keep order


def write_crawl(result: CrawlResult, out_dir: Path) -> tuple[Path, Path]:
    """Persist raw records and relationships as JSONL (the bronze layer)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    rels_path = out_dir / "relationships.jsonl"
    with records_path.open("w", encoding="utf-8") as fh:
        for record in result.records.values():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with rels_path.open("w", encoding="utf-8") as fh:
        for child, parent, rel_type in sorted(result.relationships):
            fh.write(json.dumps(
                {"child_lei": child, "parent_lei": parent, "relationship_type": rel_type}
            ) + "\n")
    return records_path, rels_path
