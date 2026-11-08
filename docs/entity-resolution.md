# Entity resolution

**Problem.** Only a fraction of sanctions records carry an identifier that links them to a
company register. Everything else must be matched on names and countries — across scripts
(Cyrillic vs Latin), legal-form conventions (ПАО / PJSC / Public Joint Stock Company) and
look-alike names ("Atlant LLC" — there are hundreds).

## Pipeline

```
companies ──► identifier links (LEI, registration number + country)   ── confidence 1.0
     │                         │ resolved companies skip the rest
     ▼
 normalize ──► block (inverted index on informative tokens) ──► score ──► decide
 (translit,     (tokens in >300 sanctions records are         (rapidfuzz    │
  accents,       too common to block on)                        + country)   │
  legal forms)                                                              ▼
            score ≥ 0.97 and multi-token name ──► FUZZY_NAME match
            0.90 ≤ score, or single-token name ─► Claude adjudicates (if enabled)
                                                  └─ or on failure ──► review queue
            score < 0.90 ───────────────────────► ignored
```

Design rules:
- **Single-token names never auto-match.** After normalization, "ООО Атлант" is just
  `atlant`. Distinctive brands (Sberbank, Severstal) and generic names (Atlant, Aspect) look
  identical to a string matcher, so both go to review.
- **The LLM can't silently decide on error.** Refusals, rate limits and network failures send
  the pair to the review queue; auth/config errors fail the run. Decisions are cached in DuckDB
  by `(lei, sanction_id, model)` so reruns are free and reproducible.
- **Only identifiers are certain.** Name-based confidence is capped at 0.99 — even when the LLM
  says 1.0.

## Evaluation

`supplygraph build-eval-set` / `supplygraph evaluate-er` (gold set committed under `eval/`).

**Labels come from identifiers, never from names.** A company is a positive for a sanctions
record if that record lists its LEI, or its registration number in the same country — the same
SQL as production. The matcher never sees identifiers, so it must recover those links from
names and countries alone.

**Population (917 companies):** every GLEIF entity whose LEI is on a sanctions record, plus
entities sampled from high-exposure jurisdictions (RU, AE, CY, BY) and ordinary supply-chain
countries (US, DE, CN). Split dev/test by hash of LEI. **Thresholds are tuned on dev; numbers
below are test.**

### Results (test: 125 positives, 318 negatives)

| System | Precision | Recall | False-alarm rate |
|---|---:|---:|---:|
| **Auto-match only (production)** | **0.969** | **0.760** | **0.63 %** |
| …if every review-band item were accepted | 0.768 | 0.872 | 9.1 % |

Review queue: **44 items (10 % of companies), containing 14 true matches** — a perfect
reviewer (human or LLM) would lift recall to 0.872 at auto-match precision.

### Ablations (same thresholds, test split)

| Change | Precision | Recall | False alarms | Takeaway |
|---|---:|---:|---:|---|
| Production config | 0.969 | 0.760 | 0.63 % | |
| Keep legal forms | 0.755 | 0.840 | 10.1 % | Legal-form stripping is the single most important step |
| Allow single-token auto-match | 0.901 | 0.872 | 2.5 % | 4× the false alarms for +11 pts recall → route to review instead |
| No transliteration | 0.979 | 0.728 | 0.31 % | Transliteration buys +3 pts recall |
| No country signal | 0.949 | 0.744 | 0.63 % | Country agreement helps both precision and recall |

### What we learned building it

1. **The first evaluation was misleading.** With only sanctioned companies in the gold set,
   precision looked like 95 % — it never measured false alarms on clean companies, the error
   that dominates real screening. Adding negatives dropped precision to 35 % at low thresholds.
2. **Then the negatives were contaminated.** Many "false alarms" (Mosenergo, Rosgosstrakh,
   QIWI Bank) were genuinely sanctioned companies whose records lack an LEI. Matching GLEIF's
   registration number (OGRN) against sanctions identifiers relabelled 22 of them objectively —
   and became a production feature (`REGISTRATION_NUMBER`).
3. **Bugs found by reading false alarms, not by tests:** "AG Capital" matched "RT-Capital"
   (German *AG* stripped + subset scoring); "Siemens Aktiengesellschaft" looked like
   "EK-Company Aktiengesellschaft" (long legal forms not stripped). Each now has a regression test.

### Known limitations

- **Chinese-script names find no candidates** (9 test misses). Sanctions lists use English
  names (e.g. CATL = 宁德时代), so this needs translation, not transliteration — a job for
  LLM- or embedding-based candidate generation.
- **Residual label noise:** both remaining test false alarms (Vympel-Communications, OZON Bank)
  have exact legal-name matches on sanctions records without identifiers — likely true matches
  the labels cannot confirm. True precision is probably higher than reported.
- **Population bias:** sanctioned entities with an LEI skew towards banks and large firms.
- **LLM hybrid not yet measured:** `evaluate-er --llm` measures Claude on the review band;
  it needs Anthropic credentials (`ANTHROPIC_API_KEY` or `ant auth login`).
