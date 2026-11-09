# 0009. The agent only proposes actions; decisions go to an append-only log

**Status:** Accepted (M6)

## Context
The agent reads text written by third parties: news headlines, filing sentences, sanctions
aliases. Any of it can contain instructions. If the agent could dismiss a sanctions match, one
planted headline could hide a sanctioned supplier. Separately, `build` rebuilds every table
(0004), so a human decision stored in a table would be erased by the next run.

## Decision
- **Proposals only.** The four ontology actions (`flag_supplier`, `request_review`,
  `confirm_match`, `dismiss_match`) are offered to the agent as *proposal* tools. A proposal
  must pass its precondition and cite evidence the agent retrieved in that session.
- **Humans decide.** An analyst approves or rejects with the CLI or the UI. Approval re-checks
  the precondition against current data.
- **Append-only log.** Proposals and decisions are events in `events.jsonl`, outside the
  warehouse. `build` folds the log into `analyst_actions`; `build` and `resolve` apply approved
  actions. `resolve` applies them before any LLM call, so a human decision is never
  re-adjudicated.

## Alternatives considered
- **Let the agent act and audit afterwards.** An audit only finds damage after it happened.
- **Keep decisions in DuckDB.** They would be erased by full-refresh builds, or would break
  idempotency.

## Consequences
- Prompt injection degrades from a data-integrity risk to, at worst, a bogus proposal in the
  review queue. The eval suite includes a planted injection to measure this.
- An approved action takes effect at the next `build` / `resolve`, not instantly.
- The log is the one file that must be backed up. In Kubernetes it lives on the PVC, whose
  `helm.sh/resource-policy: keep` stops `helm uninstall` from deleting it.
