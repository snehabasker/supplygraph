# 0014. pgvector candidate generation: measured, kept opt-in

**Status:** Accepted (M10)

## Context
Entity resolution only scores a company against sanctions records that share a whole
informative name token (token blocking, [0005](0005-entity-resolution-identifiers-first.md)).
On the test split, 5.3 % of true matches never become candidates (blocking recall 0.947), so
no threshold can recover them. Spelling variants inside tokens ("Gazpromneft" / "Gazprom Neft")
looked like a likely cause. Approximate nearest-neighbour search over character n-grams is the
standard fix.

## Decision
- `er/vectors.py` embeds each normalized sanctions name as a hashed character-trigram vector
  (256-d, deterministic, no model or API). It stores the vectors in PostgreSQL + pgvector with
  an HNSW cosine index. The index is rebuilt only when the indexed names change (fingerprint).
  Each company name adds its k = 10 nearest records to the token-blocking candidates. All
  candidates are scored by the unchanged `name_similarity`, so the thresholds keep their meaning.
- `evaluate-er --vectors` compares token-only with token + vector blocking. Each configuration
  uses its own dev-chosen thresholds, and results are reported on the test split.
- **Result: no change.** Blocking recall stays at 0.947, and test precision, recall and false
  alarms are identical (0.969 / 0.760 / 0.63 %). Every positive that blocking misses is a
  Chinese company whose GLEIF name is in Han characters only. Normalization reduces it to an
  empty string, so there is no name to embed. The one pinyin name (CATL, "ning de shi dai …")
  has no lexical overlap with its English sanctions name ("Contemporary Amperex Technology").
- Vector blocking therefore stays **opt-in and evaluation-only**. Production `resolve` keeps
  token blocking and takes no Postgres dependency.

## Rejected
- **Enabling it in production anyway.** It adds a database and an index build per run for no
  measured gain.
- **Neural sentence embeddings now.** A multilingual model could place 宁德时代 near "CATL".
  `name_similarity` would still score that pair near 0, though, so it would also need a
  cross-lingual scorer and re-tuned thresholds. That's a separate project, and the pgvector
  index is ready for those vectors.

## Consequences
- The gap is now diagnosed, not guessed: it is translation, not spelling. The next step for ER
  recall is cross-lingual candidates plus scoring, measured on the same split.
- `docker compose up -d --wait pgvector` adds an optional service. The integration test is
  gated like the Neo4j one.
