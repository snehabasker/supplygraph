"""Resolve supplier names from filings to GLEIF legal entities (LEIs).

GLEIF's name search is loose: "SK hynix" returns SK hynix America and several SK Hynix ETFs
before the Korean parent, and "Samsung Electronics" returns a dozen national subsidiaries.
So the search only *generates* candidates; a candidate is accepted only if one of its names
(legal, other, transliterated) normalizes to exactly the supplier's name. Funds and branches
are never suppliers. If several entities still tie (e.g. "Panasonic Holdings Corporation" and
"Panasonic Corporation" both normalize to "panasonic") we take the one the others roll up to;
otherwise the name stays unresolved rather than guessing.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from supplygraph.er.normalize import LEGAL_FORM_TOKENS, normalize, strip_accents, transliterate
from supplygraph.ingest.gleif import parse_record

log = logging.getLogger(__name__)

RESOLVED = "resolved"
NOT_FOUND = "not_found"
AMBIGUOUS = "ambiguous"

_EXCLUDED_CATEGORIES = {"FUND", "BRANCH"}


class GleifSearch(Protocol):
    def search_fulltext(self, query: str, page_size: int = 20) -> list[dict]: ...
    def ultimate_parent(self, lei: str) -> dict | None: ...


@dataclass(frozen=True)
class Resolution:
    name: str
    status: str
    lei: str | None = None
    legal_name: str | None = None
    candidates: int = 0


LEGAL_FORM_MISMATCH = "legal_form_mismatch"

# Abbreviations that mean the same legal form ("Co., Ltd." == "Company Limited").
_CANONICAL_FORM = {"co": "company", "ltd": "limited", "corp": "corporation",
                   "inc": "incorporated", "cos": "companies"}


def _readable(name: str) -> bool:
    """Normalization only understands Latin and Cyrillic: in "SAMSUNG ELECTRONICS ΕΛΛΑΣ ..."
    the Greek words vanish and the Greek subsidiary would look like the Korean parent."""
    return all(unicodedata.name(ch, "").startswith(("LATIN", "CYRILLIC"))
               for ch in name if ch.isalpha())


def full_key(name: str) -> str:
    """Name *including* legal form and single letters ("GLOBALFOUNDRIES U.S. Inc." keeps its
    "u s"), with abbreviations canonicalized."""
    words = re.sub(r"[^0-9a-z]+", " ", strip_accents(transliterate(name))).split()
    return " ".join(_CANONICAL_FORM.get(w, w) for w in words)


def record_names(record: dict) -> list[str]:
    row = parse_record(record)
    names = [n for n in [row["legal_name"], *(row["other_names"] or "").split(";")] if n]
    return [n for n in names if _readable(n)]


def exact_candidates(name: str, records: list[dict]) -> list[dict]:
    target = normalize(name)
    if not target:
        return []
    out = []
    for record in records:
        row = parse_record(record)
        if row["category"] in _EXCLUDED_CATEGORIES:
            continue
        if any(normalize(n) == target for n in record_names(record)):
            out.append(record)
    active = [r for r in out if parse_record(r)["entity_status"] == "ACTIVE"]
    return active or out


def form_tokens(name: str) -> set[str]:
    return {w for w in full_key(name).split() if w in LEGAL_FORM_TOKENS and w != "the"}


def states_legal_form(name: str) -> bool:
    return normalize(name) != normalize(name, drop_legal_forms=False)


def resolve_name(api: GleifSearch, name: str) -> tuple[Resolution, dict | None]:
    found: dict[str, dict] = {}
    for query in dict.fromkeys([normalize(name) or name, name]):
        for record in api.search_fulltext(query):
            found.setdefault(record["id"], record)
    matches = {r["id"]: r for r in exact_candidates(name, list(found.values()))}
    if matches and states_legal_form(name):
        # "Contemporary Amperex Technology Co. Limited" must not resolve to the German
        # "Contemporary Amperex Technology AG": same words, different legal entity. Prefer
        # the exact full name; else accept a fuller form of the same one ("Company" ->
        # "Company Limited"); a conflicting form is a mismatch.
        key, forms = full_key(name), form_tokens(name)
        exact = {lei: r for lei, r in matches.items()
                 if any(full_key(n) == key for n in record_names(r))}
        compatible = {lei: r for lei, r in matches.items()
                      if any(forms <= form_tokens(n) for n in record_names(r))}
        if not (exact or compatible):
            return Resolution(name, LEGAL_FORM_MISMATCH, candidates=len(matches)), None
        matches = exact or compatible
    if not matches:
        return Resolution(name, NOT_FOUND, candidates=len(found)), None
    if len(matches) > 1:
        # Prefer the entity the others roll up to (group parent), if that is unambiguous.
        tops = set()
        for lei in matches:
            parent = api.ultimate_parent(lei)
            tops.add(parent["id"] if parent else lei)
        if len(tops) == 1 and (top := tops.pop()) in matches:
            matches = {top: matches[top]}
        else:
            log.info("Ambiguous supplier name %r: %s", name, sorted(matches))
            return Resolution(name, AMBIGUOUS, candidates=len(matches)), None
    (lei, record), = matches.items()
    return Resolution(name, RESOLVED, lei, parse_record(record)["legal_name"], 1), record


# Bump when the resolution rules change, so cached answers from old rules are discarded.
RESOLVER_VERSION = 3


class ResolutionCache:
    """name -> resolution, persisted as JSON so reruns don't re-query GLEIF."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("version") == RESOLVER_VERSION:
                self._data = stored["names"]

    def get(self, name: str) -> tuple[Resolution, dict | None] | None:
        hit = self._data.get(normalize(name))
        if hit is None:
            return None
        return Resolution(**hit["resolution"]), hit.get("record")

    def put(self, name: str, resolution: Resolution, record: dict | None) -> None:
        self._data[normalize(name)] = {"resolution": asdict(resolution), "record": record}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": RESOLVER_VERSION, "names": self._data}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def resolve_names(
    api: GleifSearch, names: list[str], cache: ResolutionCache
) -> dict[str, tuple[Resolution, dict | None]]:
    out = {}
    for name in dict.fromkeys(names):
        hit = cache.get(name)
        if hit is None:
            hit = resolve_name(api, name)
            cache.put(name, *hit)
        out[name] = hit
    cache.save()
    return out
