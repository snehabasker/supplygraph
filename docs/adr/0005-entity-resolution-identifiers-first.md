# 0005. Entity resolution: identifiers first, dev-tuned thresholds, LLM only for the review band

**Status:** Accepted (M3)

## Context
Matching companies to sanctions records has two error costs. A missed match hides exposure. A
false match accuses a company. Real screening is dominated by false alarms on clean companies.
An early evaluation without negatives showed 95 % precision; adding negatives dropped it to
35 % at low thresholds.

## Decision
1. **Identifiers first.** An LEI, or a registration number in the same country, is a match at
   confidence 1.0, and the company skips fuzzy matching.
2. **Fuzzy matching after that.** Transliteration, legal-form stripping, blocking and rapidfuzz
   with a country signal. Single-token names never auto-match. Name-based confidence is capped
   at 0.99.
3. **Thresholds are chosen on the dev split and reported on the test split.**
   `auto_match = 0.97` is the lowest threshold with ≥ 95 % dev precision.
4. **The review band** (0.90–0.97, or any single-token name) goes to Claude when enabled, and
   otherwise to a human review queue. An LLM failure sends the pair to review; it never decides
   silently.

## Alternatives considered
- **LLM for every candidate.** Expensive, and it would hide the measurable baseline.
- **A learned matcher.** About 250 labelled positives is too few to beat a tuned rule plus a
  reviewer.

## Consequences
Test results at the production thresholds: precision 0.969, recall 0.760, false-alarm rate
0.63 % ([entity-resolution.md](../entity-resolution.md)). Any change to `er/` must be followed
by `evaluate-er`. Analyst decisions (0009) override the matcher for their pair and are never
re-adjudicated.
