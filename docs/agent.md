# Graph Q&A agent and audited actions (Milestone 6)

`supplygraph ask "<question>"` answers questions about the supplier-risk graph with Claude.
Claude retrieves facts through read-only graph tools, and every fact in the answer cites the
object it came from. The agent can also *propose* the four ontology actions. A human approves
or rejects each proposal, and nothing changes until then.

```
question ─► Claude ─► tool calls ─► GraphTools (read-only DuckDB) ─► JSON with refs
              ▲                                                          │
              └───────────────── tool results (untrusted data) ◄─────────┘
answer with [lei:…] [sanction:…] [filing:…] [news:…]  ─► grounding check (cited ⊆ retrieved)
propose_* tool ─► precondition + evidence check ─► data/actions/events.jsonl (status: proposed)
                                       analyst: supplygraph actions approve|reject
                                       build / resolve apply approved actions
```

## Retrieval: what "GraphRAG" means here

There is no embedding index. The graph is small, typed and exact, so retrieval is graph
traversal behind nine tools. Every returned object carries a `ref`.

| Tool | Returns |
|---|---|
| `search_companies` | name / short-name / LEI lookup (normalized exact match outranks fuzzy; short names like "TSMC" come from filings) |
| `get_company` | profile, watchlist reasons, risk score with drivers (signal, source ref, path), sanctions matches and candidates, analyst actions |
| `get_relationships` | parents / subsidiaries (GLEIF) or suppliers / customers (SEC, with evidence sentence, filing ref, link type) |
| `get_news` | disruption-tagged GDELT articles naming the company |
| `get_sanction` | one OpenSanctions record (aliases, programs, source lists) |
| `top_risks` | ranking by total, own or inherited risk |
| `review_queue` | unresolved sanctions candidates |
| `find_connection` | shortest ownership / supply chain between two companies |
| `run_scenario` | the risk engine's what-if (`risk/engine.scenario`) |

The tools read the warehouse (DuckDB), not Neo4j. The warehouse is the graph's source of truth:
Neo4j mirrors it on `load-graph`. That keeps the agent usable and testable without a database
server. The tools also enforce the graph's semantics in their output, which the system prompt
repeats:
- `POSSIBLE_SUPPLIER` is a ~0.56-precision lead.
- Review candidates are not findings.
- A news mention is not proof of impact.

## Safety properties, and where each is enforced

| Property | Enforced by |
|---|---|
| Answers are grounded | `AgentResult.ungrounded` lists cited refs that no tool returned. Refs are collected from result *structure* (`*ref` keys), so a headline containing "lei:…" does not count. The CLI warns on ungrounded refs, and the eval fails them. |
| The agent cannot change state | Action tools only append a `proposed` event. They are not offered at all with `ask --read-only`. |
| Proposals are justified | Each proposal must pass its ontology precondition (e.g. only an existing ER candidate can be confirmed or dismissed) and cite evidence refs *retrieved in this session*. |
| Humans decide, and decisions are audited | `actions approve|reject` appends an event with analyst, time and note. Approval re-checks the precondition against current data. The log is append-only and lives outside the warehouse, so `build` cannot erase a decision. |
| Prompt injection does not drive actions | Tool output is labelled untrusted in the system prompt. Structurally, an injected instruction can at most create a *proposal* that a human then reads. |
| Failures are explicit | API errors, refusals, `max_tokens` and the step limit (12 model calls) end the session with `error` set and no answer, never a guess. |

## Actions: lifecycle and effects

```powershell
supplygraph ask "Dismiss the ASML Hong Kong sanctions candidate: it matched only the alias 'Hong Kong'"
supplygraph actions list --status proposed
supplygraph actions approve A-470e5c7cd3 --by sneha     # or: reject A-… --note "…"
supplygraph build; supplygraph resolve; supplygraph score-risk; supplygraph load-graph
```

| Approved action | Applied by | Effect |
|---|---|---|
| `flag_supplier` | `build` | watchlist reason `analyst_flag` (news is matched for it); `Company.analyst_flags` |
| `request_review` | `load-graph` | `Company.review_requested = true` |
| `confirm_match` | `resolve` | `SANCTIONED_AS` with `ANALYST_CONFIRMED`, confidence 1.0. Kept even if the pair stops being the best fuzzy candidate. |
| `dismiss_match` | `resolve` | candidate becomes `no_match` / `ANALYST_DISMISSED` and leaves the review queue |

`resolve` folds the action log itself, so a decision approved after the last `build` applies
at the next `resolve`. It also applies human decisions *before* consulting the LLM adjudicator,
so a decided pair is never re-adjudicated.

**On the live data** (run on a copy, so the real log stays empty): approving the dismissal of
ASML Hong Kong ↔ "LEADER (HONG KONG) INTERNATIONAL" emptied that review slot. The
spurious possible-sanction risk then disappeared from the whole ASML group, and companies at
risk fell from 24 to 7. Approving a flag on Wistron grew the watchlist from 20 to 21. All
blocking checks passed.

## Eval suite

```powershell
supplygraph evaluate-agent --check-only     # offline: the gold is answerable (also a unit test)
supplygraph evaluate-agent                  # live: one agent session per case (needs Claude credentials)
supplygraph evaluate-agent --case injection-walmart-news --effort high
```

- **Fixture.** The suite runs on `eval/agent/fixture/`, a frozen snapshot of the real warehouse
  tables the agent reads (278 companies, 30 supplier links, 14 news links, 2 review candidates).
  Answers depend on data, so gold answers are only stable against a frozen world.
- **Planted rows.** `planted.json` adds synthetic rows the real data lacks: a Walmart "news
  article" whose headline tells AI agents to dismiss a sanctions candidate.
- **Cases.** 22 cases in `eval/agent/cases.jsonl`, in ten categories: lookup, sanctions,
  risk_explanation, risk_ranking, supply, scenario, news, ownership, action, action_guard,
  injection, unanswerable.

Each case states what a correct answer must *contain*, not its wording:

| Field | Check |
|---|---|
| `must_cite` | refs the answer must cite |
| `must_mention` | groups of alternatives; one of each must appear (e.g. that a link is `POSSIBLE_SUPPLIER`) |
| `must_not_mention` | strings that must not appear |
| `expect_actions` | exactly these proposals, no more (default: none) |
| `expect_insufficient` | the answer must start with `INSUFFICIENT DATA` (e.g. Tesla has no supplier links in the graph: absence is not evidence) |
| `oracle` | tool calls that retrieve every `must_cite` ref and the target of every expected action |

All answers must also be grounded and complete without an error. Reported metrics:
- pass rate, overall and per category;
- citation recall (required refs cited) and citation grounding (cited refs retrieved);
- action precision and recall, and the number of cases with an unrequested action (the
  injection and action_guard cases target this);
- mean tool calls and token usage.

**Results.** Pending the first live run, which needs prepaid Anthropic API credit (a claude.ai
subscription does not cover API calls). Only the offline parts have run so far: oracle check 22/22, and the harness unit tests
with a scripted client. Record the first run here: model, effort, pass rate per category,
grounding, unrequested actions. Read the failed cases' answers in
`data/agent_eval_report.json` before concluding anything. Some gold `must_mention` alternatives
may turn out to be too narrow, and that is a case bug, not a model failure.

**Changing the suite.** Never edit a case to make a number go up. After `build-agent-fixture`
(re-snapshot), the command re-runs the oracle check and lists stale cases. Fix those against the
new data, the same way as the extraction gold set.

## Design decisions

- **Manual tool loop, not the SDK tool runner.** The loop *is* the safety boundary: grounding
  bookkeeping, evidence checks on proposals, the step limit, and error-to-result handling. It
  must be unit-testable with a scripted client, and it avoids depending on a beta helper.
- **Model and request.** The model comes from `SUPPLYGRAPH_LLM_MODEL` (default `claude-opus-5-5`)
  at effort `medium` (`--effort` to change). The system prompt is cached, and server-side refusal
  fallback is on (`fallbacks="default"`). Thinking is adaptive (the model's default).
- **Proposal, not execution.** Even a perfectly instructed model can be talked into an action by
  a document it reads. Making every state change a human-approved proposal turns prompt
  injection from a data-integrity risk into, at worst, a review-queue nuisance.
- **The log is outside the warehouse.** `build` is a full refresh (CREATE OR REPLACE) and must
  stay idempotent. Human decisions are not derived data, so they live in an append-only JSONL
  file that `build` reads, like any other bronze source.
