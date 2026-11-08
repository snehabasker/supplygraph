"""Link GDELT disruption events to watchlist companies.

GDELT names organizations the way journalists do ("nvidia", "samsung electronics", "tsmc"),
so a company is matched on any of its names after the same normalization entity resolution
uses (legal forms stripped): its GLEIF legal and other names, plus the short names its peers'
filings define for it ("TSMC" for Taiwan Semiconductor Manufacturing Company Limited).

Matching is exact on the normalized form — no fuzzy scoring — and only watchlist companies
take part: a crawl of 25 subsidiaries per seed contains plenty of generic names ("Atlas
GmbH") that would match unrelated news. If one name maps to several watchlist companies
(BASF SE and BASF Corporation are both "basf"), the group parent wins when the ownership
table says so; otherwise the name is ambiguous and skipped.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import duckdb

from supplygraph.er.normalize import normalize

MIN_NAME_CHARS = 3


def watchlist_names(
    con: duckdb.DuckDBPyConnection, aliases: Mapping[str, str]
) -> dict[str, set[str]]:
    """normalized name -> watchlist LEIs that carry it."""
    rows = con.execute("""
        SELECT c.lei, c.legal_name, c.other_names
        FROM companies c SEMI JOIN watchlist w USING (lei)
    """).fetchall()
    names: dict[str, set[str]] = defaultdict(set)
    lei_names: dict[str, set[str]] = defaultdict(set)
    for lei, legal_name, other_names in rows:
        for name in [legal_name, *(other_names or "").split(";")]:
            norm = normalize(name)
            if len(norm) >= MIN_NAME_CHARS:
                names[norm].add(lei)
                lei_names[lei].add(norm)
    # Supplier names as written in filings resolve to LEIs too ("Samsung Electronics").
    by_supplier_name: dict[str, set[str]] = defaultdict(set)
    for supplier_name, lei in con.execute(
        "SELECT DISTINCT supplier_name, supplier_lei FROM supplier_mentions "
        "WHERE supplier_lei IS NOT NULL"
    ).fetchall():
        by_supplier_name[normalize(supplier_name)].add(lei)
    watch = {lei for lei, _, _ in rows}
    for alias, full_name in aliases.items():
        norm_alias, norm_full = normalize(alias), normalize(full_name)
        if len(norm_alias) < MIN_NAME_CHARS:
            continue
        targets = (names.get(norm_full, set()) | by_supplier_name.get(norm_full, set())) & watch
        names[norm_alias] |= targets
    return {k: v for k, v in names.items() if v}


def _parents(con: duckdb.DuckDBPyConnection) -> dict[str, set[str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    for child, parent in con.execute("SELECT child_lei, parent_lei FROM ownership").fetchall():
        parents[child].add(parent)
    return parents


def disambiguate(leis: set[str], parents: Mapping[str, set[str]]) -> str | None:
    if len(leis) == 1:
        return next(iter(leis))
    tops = {lei for lei in leis if not (parents.get(lei, set()) & leis)}
    return next(iter(tops)) if len(tops) == 1 else None


def match_events(
    con: duckdb.DuckDBPyConnection, events: list[dict], aliases: Mapping[str, str]
) -> tuple[list[list[str]], dict[str, int]]:
    """Returns ([lei, event_id, matched_name] rows, stats)."""
    names = watchlist_names(con, aliases)
    parents = _parents(con)
    resolved: dict[str, str | None] = {norm: disambiguate(leis, parents)
                                       for norm, leis in names.items()}
    links: dict[tuple[str, str], str] = {}
    ambiguous_hits = 0
    for event in events:
        for org in event["organizations"]:
            norm = normalize(org)
            if norm not in resolved:
                continue
            lei = resolved[norm]
            if lei is None:
                ambiguous_hits += 1
                continue
            links.setdefault((lei, event["id"]), org)
    rows = [[lei, event_id, org] for (lei, event_id), org in sorted(links.items())]
    stats = {"watchlist_names": len(names),
             "ambiguous_names": sum(v is None for v in resolved.values()),
             "ambiguous_hits": ambiguous_hits}
    return rows, stats
