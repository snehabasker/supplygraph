# News story clustering

`dedupe_syndicated` merges only identical headlines. Outlets rewrite the same story, so
`build` follows it with `news_cluster.merge_stories`: **unsupervised** story clustering.

**Method.** Title tokens (stop-words dropped) -> sublinear TF-IDF (IDF over the batch) ->
cosine similarity via an inverted index -> single-linkage clusters (union-find) when
cosine >= threshold, the events are within 36 h, and they share >= 2 content words. No
labels are used to form clusters. Each cluster collapses to its earliest event; organizations
and event types are unioned and `copies` sums (so `NewsEvent.copies` now counts syndicated
copies *and* rewritten headlines of one story). No scikit-learn: the code is dependency-free.

## Evaluation (`supplygraph evaluate-news-clusters`)

Gold: 120 headline pairs (`eval/news/gold_pairs.csv`, snapshot in `eval/news/events.jsonl`),
hand-labelled "same story?" and split dev/test by hash of the pair. The pairs are a
**stratified sample of candidate pairs by similarity**, not the natural mix, so precision and
recall hold *within that pool*; they are not corpus-wide rates. Because candidates are pairs
that exact-title dedupe missed, the exact-title baseline scores 0 recall by construction.
Threshold: the best dev F1 among thresholds with dev precision >= 0.95 (a wrong merge drops an
event from the graph; a miss only leaves a duplicate). Test pairs: 47 (37 same).

| System (test) | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Exact title (existing dedupe) | - | 0.000 | 0.000 | 0.000 |
| Token Jaccard | 0.50 | 0.933 | 0.378 | 0.538 |
| **TF-IDF single linkage** | **0.50** | **0.900** | **0.730** | **0.806** |

Best-F1 on dev alone would pick 0.40 (dev P 0.90 / R 0.98) and add false merges
(Namibia/Japan earthquakes, Canada/China sanctions), hence the precision floor.

**Real data (2026-11-07/08 GDELT):** 4,053 raw records -> 2,508 after exact-title dedupe ->
**2,094 stories** (-414, -16.5 %). `company_news` unchanged (14 links); total copies conserved.

## Known errors (test split)

- **Template headlines, different place:** "earthquake strikes Namibia" merged with "...strikes
  Japan" (2 of 3 false merges). Needs entity/number agreement, not more TF-IDF.
- **Reaction vs. report:** "Trump names Miller" merged with "El-Sayed criticizes Miller's
  appointment". The labelling convention is *same incident*; reactions are labelled different.
- **Misses:** paraphrases with little word overlap ("Houston limits ICE cooperation" vs
  "Whitmire directs HPD to reduce cooperation with ICE").
- **Label noise:** follow-ups with updated numbers (toll two vs three) are labelled same story.
- 120 pairs is small; read the intervals as wide.
