# 0013. Evaluation runs are tracked in MLflow; prompts are versioned; the LLM judge is advisory

**Status:** Accepted (M10)

## Context
Every model-facing component has its own evaluation (`evaluate-er`, `-extraction`, `-risk`,
`-news-clusters`, `-agent`), and each writes a JSON report. A report shows the current numbers.
It doesn't record what the numbers were before a change, or which prompt, model, thresholds and
commit produced them. Prompts made this worse. The LLM extraction cache was keyed on a
hand-bumped `PROMPT_VERSION`, but nothing checked that anyone bumped it. The agent's prompt
(system text plus tool descriptions) had no version at all.

The agent eval's rule-based grader checks citations, grounding, required and forbidden strings,
and actions. It can't check whether the sentence around a valid citation is true. "TSMC is
headquartered in Japan [lei:TSMC]" passes every rule.

## Decision
- **MLflow tracking, opt-in.** `--track` on every `evaluate-*` command logs:
  - params: git commit, model, prompt versions, thresholds or weights;
  - every numeric leaf of the report as a metric;
  - threshold sweeps as stepped series;
  - the full report as an artifact.

  The default store is local SQLite under the data dir. `MLFLOW_TRACKING_URI` points at a
  shared server instead. MLflow is an optional extra, and only `tracking.py` imports it.
- **Prompt registry and lock.** Each LLM prompt declares a `PROMPT_VERSION`. `prompts.py`
  fingerprints the prompt's text, tool specs and output schema. `prompts.lock.json` pins
  (version, fingerprint), and a unit test fails when content changes without a version bump.
  `supplygraph prompts --update` refreshes the lock after a bump.
- **LLM-as-judge, advisory.** `evaluate-agent --judge` asks a second Claude call whether every
  claim in the answer is supported by the evidence the agent retrieved. The evidence is the
  agent's own read-only tool calls, replayed on the fixture. The verdict is structured and
  scored 1–5. The pass rate stays rule-based. The judge is reported next to it, with its
  agreement with the rules (Cohen's kappa) and the list of cases where the two disagree.

## Rejected
- **Judge as the pass/fail gate.** An unvalidated judge would decide whether changes ship. It
  should first be calibrated against human labels on its disagreements with the rules.
- **Weights & Biases / a hosted tracker.** That needs an account and network access. MLflow
  runs locally with no service.
- **Model registry.** Nothing here is a trained artifact yet. The link-prediction model ships
  only if it beats the popularity baseline (docs/risk.md), and none does. The registry becomes
  relevant when one does.

## Consequences
- Prompt changes now require a version bump plus `prompts --update`. That's one more step, and
  intended.
- The judge doubles the LLM calls of a live agent eval.
- The judge has not run live yet. Like the agent eval, it needs prepaid API credit.
