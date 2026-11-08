"""Supplier-extraction evaluation against a hand-labelled gold set.

eval/extraction/passages.jsonl  candidate passages from the latest filings (frozen, so the
                                evaluation is reproducible offline and in CI)
eval/extraction/aliases.json    short names learned from those filings
eval/extraction/gold.csv        filer_cik, supplier, aliases, label (supplier|ambiguous), note

Unit of evaluation: a (filer, supplier entity) relationship — the edge that would be loaded
into the graph — not individual mentions. A prediction matches a gold supplier if its
normalized name equals the supplier's or one of its aliases. Predictions of `ambiguous` gold
rows (e.g. a joint venture that also provides services) count neither way.

Gold labels were made by reading every candidate passage; they describe what the passage
*states*, not outside knowledge (Intel uses TSMC — that is gold because Intel's 10-K says so).
Recall is therefore relative to the candidate passages: a supplier mentioned only in a
sentence the candidate filter skipped is invisible to both the extractor and the gold set.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from supplygraph.er.normalize import normalize
from supplygraph.extract.suppliers import Extractor, FilerContext, Mention
from supplygraph.extract.text import Passage

SUPPLIER = "supplier"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class GoldRow:
    filer_cik: str
    supplier: str
    aliases: tuple[str, ...]
    label: str
    note: str

    def name_keys(self) -> set[str]:
        return {k for k in (normalize(n) for n in (self.supplier, *self.aliases)) if k}


def write_eval_set(out_dir: Path, aliases: dict[str, str], contexts: dict[str, FilerContext],
                   passages: dict[str, list[Passage]]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aliases.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    n = 0
    with (out_dir / "passages.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for cik, items in passages.items():
            ctx = contexts[cik]
            for p in items:
                fh.write(json.dumps({"filer": asdict(ctx), **asdict(p)},
                                    ensure_ascii=False) + "\n")
                n += 1
    return n


def load_eval_set(eval_dir: Path) -> tuple[dict[str, str], list[tuple[FilerContext, Passage]],
                                           list[GoldRow]]:
    aliases = json.loads((eval_dir / "aliases.json").read_text(encoding="utf-8"))
    items = []
    with (eval_dir / "passages.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            f = row.pop("filer")
            items.append((FilerContext(f["cik"], f["name"], tuple(f["self_names"])),
                          Passage(**row)))
    with (eval_dir / "gold.csv").open(encoding="utf-8", newline="") as fh:
        gold = [GoldRow(r["filer_cik"], r["supplier"],
                        tuple(a for a in r["aliases"].split(";") if a), r["label"], r["note"])
                for r in csv.DictReader(fh)]
    return aliases, items, gold


@dataclass
class ExtractionMetrics:
    tp: int
    fp: int
    fn: int
    failed_passages: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "failed_passages": self.failed_passages,
                "precision": round(self.precision, 3), "recall": round(self.recall, 3),
                "f1": round(self.f1, 3)}


def score(mentions: list[Mention], gold: list[GoldRow], failed: int = 0) -> dict:
    by_filer: dict[str, list[GoldRow]] = defaultdict(list)
    for g in gold:
        by_filer[g.filer_cik].append(g)

    found: set[tuple[str, str]] = set()  # (cik, gold supplier) that were predicted
    false_pos: dict[tuple[str, str], Mention] = {}
    for m in mentions:
        key = normalize(m.supplier_name)
        written = normalize(m.name_as_written)
        hit = next((g for g in by_filer.get(m.filer_cik, [])
                    if key in g.name_keys() or written in g.name_keys()), None)
        if hit is None:
            false_pos.setdefault((m.filer_cik, key), m)
        elif hit.label == SUPPLIER:
            found.add((hit.filer_cik, hit.supplier))
    positives = [g for g in gold if g.label == SUPPLIER]
    missed = [g for g in positives if (g.filer_cik, g.supplier) not in found]
    metrics = ExtractionMetrics(len(found), len(false_pos), len(missed), failed)
    return {
        **metrics.as_dict(),
        "false_positives": [{"filer_cik": cik, "supplier": m.supplier_name,
                             "evidence": m.evidence[:300]}
                            for (cik, _), m in sorted(false_pos.items())],
        "missed": [{"filer_cik": g.filer_cik, "supplier": g.supplier, "note": g.note}
                   for g in missed],
    }


def evaluate(eval_dir: Path, extractors: dict[str, Extractor]) -> dict:
    _, items, gold = load_eval_set(eval_dir)
    report = {
        "passages": len(items),
        "filers": len({ctx.cik for ctx, _ in items}),
        "gold_suppliers": sum(g.label == SUPPLIER for g in gold),
        "gold_ambiguous": sum(g.label == AMBIGUOUS for g in gold),
        "systems": {},
    }
    for name, extractor in extractors.items():
        mentions, failed = [], 0
        for ctx, passage in items:
            found = extractor.extract(passage, ctx)
            if found is None:
                failed += 1
            else:
                mentions.extend(found)
        report["systems"][name] = score(mentions, gold, failed)
    return report
