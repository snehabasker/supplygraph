"""Filings -> passages -> supplier mentions -> LEIs (the `extract-suppliers` step).

Outputs (bronze, under data/raw):
  sec/aliases.json              short names learned from all filings ("TSMC" -> full name)
  sec/supplier_mentions.jsonl   one row per grounded mention, with its LEI resolution
  sec/name_resolutions.json     GLEIF lookup cache
  gleif/suppliers/*.jsonl       GLEIF records (+ parents) of resolved suppliers
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from supplygraph.extract.link import RESOLVED, ResolutionCache, resolve_names
from supplygraph.extract.suppliers import Extractor, FilerContext, Mention
from supplygraph.extract.text import (
    Passage,
    aliases_excluding_filer,
    candidate_passages,
    html_to_text,
    learn_aliases,
    merge_aliases,
)
from supplygraph.ingest import gleif
from supplygraph.ingest.sec import FilingRef

log = logging.getLogger(__name__)


def load_texts(sec_dir: Path, refs: list[FilingRef]) -> dict[str, str]:
    return {ref.cik: html_to_text((sec_dir / ref.path).read_text(encoding="utf-8",
                                                                  errors="replace"))
            for ref in refs}


def filer_context(ref: FilingRef) -> FilerContext:
    return FilerContext(ref.cik, ref.filer_name, (ref.entity_name,))


def build_passages(
    refs: list[FilingRef], texts: dict[str, str]
) -> tuple[dict[str, str], dict[str, list[Passage]]]:
    aliases = merge_aliases(learn_aliases(texts[ref.cik]) for ref in refs)
    passages = {}
    for ref in refs:
        ctx = filer_context(ref)
        passages[ref.cik] = candidate_passages(
            ref.cik, texts[ref.cik], aliases_excluding_filer(aliases, ctx.all_names),
            self_names=ctx.all_names,
        )
    return aliases, passages


def extract_mentions(
    extractor: Extractor, refs: list[FilingRef], passages: dict[str, list[Passage]]
) -> tuple[list[Mention], int]:
    """All grounded mentions, de-duplicated per (filer, supplier, passage); plus the number of
    passages whose extraction failed."""
    mentions: dict[tuple, Mention] = {}
    failed = 0
    for ref in refs:
        ctx = filer_context(ref)
        for passage in passages[ref.cik]:
            found = extractor.extract(passage, ctx)
            if found is None:
                failed += 1
                continue
            for m in found:
                mentions.setdefault((*m.key, m.passage_id), m)
    return list(mentions.values()), failed


def run(
    sec_dir: Path,
    supplier_gleif_dir: Path,
    refs: list[FilingRef],
    extractor: Extractor,
    api: gleif.GleifClient,
) -> dict[str, int]:
    texts = load_texts(sec_dir, refs)
    aliases, passages = build_passages(refs, texts)
    (sec_dir / "aliases.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    mentions, failed = extract_mentions(extractor, refs, passages)

    cache = ResolutionCache(sec_dir / "name_resolutions.json")
    resolutions = resolve_names(api, sorted({m.supplier_name for m in mentions}), cache)
    by_cik = {ref.cik: ref for ref in refs}
    with (sec_dir / "supplier_mentions.jsonl").open("w", encoding="utf-8") as fh:
        for m in mentions:
            res, _ = resolutions[m.supplier_name]
            ref = by_cik[m.filer_cik]
            fh.write(json.dumps({
                **asdict(m),
                "filer_lei": ref.filer_lei, "accession": ref.accession, "form": ref.form,
                "filing_date": ref.filing_date,
                "supplier_lei": res.lei, "resolution_status": res.status,
                "resolved_legal_name": res.legal_name,
            }, ensure_ascii=False) + "\n")

    # Bring resolved suppliers (and their corporate parents) into the company universe.
    supplier_leis = sorted({res.lei for res, _ in resolutions.values()
                            if res.status == RESOLVED and res.lei})
    crawl = gleif.crawl(api, supplier_leis, children_limit=0)
    gleif.write_crawl(crawl, supplier_gleif_dir)

    statuses = [res.status for res, _ in resolutions.values()]
    counts = {
        "filings": len(refs),
        "passages": sum(len(p) for p in passages.values()),
        "failed_passages": failed,
        "mentions": len(mentions),
        "supplier_names": len(resolutions),
        "resolved": statuses.count(RESOLVED),
        "unresolved": len(statuses) - statuses.count(RESOLVED),
        "supplier_entities": len(crawl.records),
    }
    log.info("supplier extraction: %s", counts)
    return counts
