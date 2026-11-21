"""Unsupervised story clustering for news events.

`dedupe_syndicated` only merges byte-identical titles. The same story is often headlined
differently by each outlet, so near-duplicates survive and inflate the event count (and, later,
risk signals). This clusters events by title similarity: TF-IDF vectors, cosine similarity,
single-linkage agglomeration (union-find) under a similarity threshold and a time window.

Dependency-free on purpose (no scikit-learn): an inverted index keeps it near-linear on
GDELT-sized batches. Unsupervised: no labels are used to form clusters; labels are used only to
*evaluate* it and to pick the threshold on the dev split (see `evaluate`).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime

DEFAULT_THRESHOLD = 0.5  # chosen on the dev split by news_cluster.evaluate
DEFAULT_MAX_GAP_HOURS = 36.0
MIN_PRECISION = 0.95  # dev-precision floor when choosing the production threshold
MIN_SHARED_TOKENS = 2  # one shared word ("Iran", "storm") is never a story

_STOP_TEXT = (
    "a an and are as at be by for from has have he her his in into is it its of on or over "
    "says say said she that the their they this to was were will with after before new news "
    "us up out about amid more than how what who why when")
STOPWORDS = frozenset(_STOP_TEXT.split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(title: str | None) -> list[str]:
    """Lowercased content words; empty for an untitled event (which never clusters)."""
    return [t for t in _TOKEN.findall((title or "").lower())
            if t not in STOPWORDS and len(t) > 1]


def _parse(ts: str | datetime | None) -> datetime | None:
    if isinstance(ts, datetime):
        return ts
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _vectors(token_lists: list[list[str]]) -> list[dict[str, float]]:
    """L2-normalised sublinear TF-IDF; IDF is computed over the batch being clustered."""
    n = len(token_lists)
    df = Counter(t for toks in token_lists for t in set(toks))
    out: list[dict[str, float]] = []
    for toks in token_lists:
        vec = {t: (1 + math.log(c)) * (math.log((1 + n) / (1 + df[t])) + 1)
               for t, c in Counter(toks).items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        out.append({t: v / norm for t, v in vec.items()})
    return out


def similar_pairs(events: list[dict], min_similarity: float,
                  max_gap_hours: float = DEFAULT_MAX_GAP_HOURS):
    """Yield (i, j, cosine) for every pair above `min_similarity` that sits in the time window
    and shares at least MIN_SHARED_TOKENS words. Untitled events never pair."""
    toks = [tokens(e.get("title")) for e in events]
    vecs = _vectors(toks)
    times = [_parse(e.get("published_at")) for e in events]
    index: dict[str, list[int]] = defaultdict(list)
    for i, vec in enumerate(vecs):
        shared: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for t, w in vec.items():
            for j in index[t]:
                shared[j] += w * vecs[j][t]
                counts[j] += 1
        for j, cos in shared.items():
            if cos < min_similarity or counts[j] < MIN_SHARED_TOKENS:
                continue
            if times[i] and times[j] and abs((times[i] - times[j]).total_seconds()) \
                    > max_gap_hours * 3600:
                continue
            yield j, i, cos
        for t in vec:
            index[t].append(i)


def cluster_stories(events: list[dict], threshold: float = DEFAULT_THRESHOLD,
                    max_gap_hours: float = DEFAULT_MAX_GAP_HOURS) -> list[list[int]]:
    """Indices of `events` grouped into stories (singletons included), earliest event first."""
    parent = list(range(len(events)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in similar_pairs(events, threshold, max_gap_hours):
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(events)):
        groups[find(i)].append(i)
    return sorted(
        (sorted(g, key=lambda k: (str(events[k].get("published_at") or ""), events[k]["id"]))
         for g in groups.values()),
        key=lambda g: g[0])


def merge_stories(events: list[dict], threshold: float = DEFAULT_THRESHOLD,
                  max_gap_hours: float = DEFAULT_MAX_GAP_HOURS) -> list[dict]:
    """Collapse each story to its earliest event, like `dedupe_syndicated`: organizations and
    event types are unioned and `copies` sums the members' copies."""
    out = []
    for group in cluster_stories(events, threshold, max_gap_hours):
        first = dict(events[group[0]])
        if len(group) > 1:
            members = [events[k] for k in group]
            first["organizations"] = list(dict.fromkeys(
                o for m in members for o in m.get("organizations") or []))
            first["event_types"] = sorted({t for m in members for t in m.get("event_types") or []})
            first["copies"] = sum(m.get("copies") or 1 for m in members)
        out.append(first)
    return out


# ------------------------------------------------------------------------------- evaluation

def candidate_pairs(events: list[dict], min_similarity: float = 0.25) -> list[tuple[int, int]]:
    """The pool a human labels: pairs similar enough that merging them is a real question."""
    return sorted((i, j) for i, j, _ in similar_pairs(events, min_similarity))


def jaccard_pairs(events: list[dict], threshold: float,
                  max_gap_hours: float = DEFAULT_MAX_GAP_HOURS) -> set[tuple[int, int]]:
    """Baseline: token-set Jaccard, no IDF weighting."""
    sets = [set(tokens(e.get("title"))) for e in events]
    out = set()
    for i, j, _ in similar_pairs(events, 0.0, max_gap_hours):
        union = sets[i] | sets[j]
        if union and len(sets[i] & sets[j]) / len(union) >= threshold:
            out.add((min(i, j), max(i, j)))
    return out


def predicted_same(events: list[dict], threshold: float) -> set[tuple[int, int]]:
    """All index pairs that land in the same cluster at `threshold` (transitive closure)."""
    pairs: set[tuple[int, int]] = set()
    for group in cluster_stories(events, threshold):
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                pairs.add((min(group[a], group[b]), max(group[a], group[b])))
    return pairs


def score_pairs(predicted: set[tuple[int, int]], gold: dict[tuple[int, int], bool]) -> dict:
    """Pair precision/recall *within the labelled pool* (see docs/news-clustering.md)."""
    tp = sum(1 for p, same in gold.items() if same and p in predicted)
    fp = sum(1 for p, same in gold.items() if not same and p in predicted)
    fn = sum(1 for p, same in gold.items() if same and p not in predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def load_gold(events_path, gold_path) -> tuple[list[dict], dict[str, dict[tuple[int, int], bool]]]:
    """The committed snapshot and labelled pairs, as index pairs, keyed by split."""
    import csv
    import json
    from pathlib import Path

    events = [json.loads(line) for line in Path(events_path).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    pos = {e["id"]: i for i, e in enumerate(events)}
    gold: dict[str, dict[tuple[int, int], bool]] = {"dev": {}, "test": {}}
    with Path(gold_path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            i, j = sorted((pos[row["id_a"]], pos[row["id_b"]]))
            gold[row["split"]][(i, j)] = row["same_story"] == "1"
    return events, gold


def exact_title_pairs(events: list[dict]) -> set[tuple[int, int]]:
    """What `dedupe_syndicated` already does: identical normalised titles."""
    seen: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(events):
        key = " ".join((e.get("title") or "").lower().split())
        if key:
            seen[key].append(i)
    return {(a, b) for g in seen.values() for a in g for b in g if a < b}


def evaluate(events_path, gold_path,
             thresholds=(0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8)) -> dict:
    """Tune the threshold on dev (best F1), report test. Baselines: exact title (the existing
    dedupe) and unweighted Jaccard, each also tuned on dev."""
    events, gold = load_gold(events_path, gold_path)

    def best(predict, grid):
        """Highest dev F1 among operating points with dev precision >= MIN_PRECISION: a wrong
        merge silently drops an event from the graph, a missed merge only leaves a duplicate."""
        scored = [(score_pairs(predict(t), gold["dev"]), t) for t in grid]
        ok = [(m["f1"], t) for m, t in scored if m["precision"] >= MIN_PRECISION]
        return max(ok or [(m["f1"], t) for m, t in scored], key=lambda s: (s[0], s[1]))[1]

    tfidf_t = best(lambda t: predicted_same(events, t), thresholds)
    jacc_grid = (0.3, 0.4, 0.5, 0.6, 0.7)
    jacc_t = best(lambda t: jaccard_pairs(events, t), jacc_grid)
    systems = {
        "exact_title": (None, exact_title_pairs(events)),
        "jaccard": (jacc_t, jaccard_pairs(events, jacc_t)),
        "tfidf_single_linkage": (tfidf_t, predicted_same(events, tfidf_t)),
    }
    report = {
        "events": len(events),
        "labelled": {s: {"pairs": len(g), "same": sum(g.values())} for s, g in gold.items()},
        "systems": {
            name: {"threshold": t, "dev": score_pairs(p, gold["dev"]),
                   "test": score_pairs(p, gold["test"])}
            for name, (t, p) in systems.items()},
        "dev_sweep": {str(t): score_pairs(predicted_same(events, t), gold["dev"])
                      for t in thresholds},
    }
    p = systems["tfidf_single_linkage"][1]
    report["test_errors"] = {
        "false_merges": [[events[i]["title"], events[j]["title"]]
                         for (i, j), same in gold["test"].items() if not same and (i, j) in p],
        "missed_merges": [[events[i]["title"], events[j]["title"]]
                          for (i, j), same in gold["test"].items() if same and (i, j) not in p],
    }
    return report
