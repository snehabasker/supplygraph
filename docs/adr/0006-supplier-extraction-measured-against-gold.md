# 0006. Supplier extraction is measured against a hand-labelled gold set

**Status:** Accepted (M4)

## Context
Annual reports name suppliers, but also customers, competitors and partners, often in the
same sentence ("TSMC, a key foundry competitor" next to "we use TSMC"). An extractor that
looks plausible on a few examples can be mostly wrong.

## Decision
- **Gold set.** 82 candidate passages from 12 filers, hand-labelled by what the passage
  *states*: 27 supplier relationships. They are committed as `eval/extraction/`, and nobody
  edits the gold to raise a score.
- **Two extractors.** A rules baseline, whose output becomes `POSSIBLE_SUPPLIER`, and Claude
  with structured output and verbatim-evidence grounding, whose output becomes `SUPPLIES` (0003).
- **Entity linking.** Name → LEI linking through GLEIF is versioned (`RESOLVER_VERSION`).

## Consequences
- The rule baseline measures precision 0.559 and recall 0.704
  ([extraction.md](../extraction.md)), so it can only create leads.
- The Claude extractor's precision is not yet measured: it needs API credit. Until it is, the
  live graph has zero asserted `SUPPLIES` edges, and every supply-based conclusion in the UI and
  agent is labelled unverified.
