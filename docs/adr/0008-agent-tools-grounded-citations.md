# 0008. The agent retrieves through typed tools and must cite what it retrieved

**Status:** Accepted (M6)

## Context
"GraphRAG" usually means embedding text chunks and retrieving by similarity. This graph is
small, typed and exact. The risk is not *finding* facts. The risk is the model asserting facts
it didn't retrieve, or blurring a lead into a finding.

## Decision
- **Typed tools, no vector index.** Nine read-only tools over the warehouse: search, profile,
  relationships, news, sanctions record, top risks, review queue, connection path, scenario.
  Every returned object carries a ref (`lei:`, `sanction:`, `filing:`, `news:`).
- **Grounding is checked in code.** The answer cites refs, and the loop records which refs the
  tools returned (from result *structure*, so a headline containing "lei:…" does not count).
  A citation outside that set is flagged as ungrounded.
- **A manual loop, not the SDK tool runner.** The loop is the safety boundary: grounding
  bookkeeping, evidence checks on proposals, a step limit, and errors that end the session
  instead of producing a guess. It has to be unit-testable with a scripted client.
- **The UI reuses the same tools** (M7), so the agent and the explorer cannot disagree about
  the data.

## Consequences
- The eval suite (22 cases on a frozen fixture) can grade answers mechanically: required
  citations, grounding, actions, abstention.
- The live eval score is still unmeasured: it needs API credit ([agent.md](../agent.md)).
- Free-text questions about content the graph doesn't hold (revenue, executives) get
  "INSUFFICIENT DATA", by design.
