# SupplyGraph ontology (v0.4)

The ontology is the contract between data pipelines, the graph, the agent and the UI.
Every source is mapped *into* these types; nothing downstream reads raw source fields.

## Object types

| Type | Key | Source | Key properties |
|---|---|---|---|
| `Company` | `lei` (ISO 17442) | GLEIF | `legal_name`, `other_names`, `registered_as`, `jurisdiction`, `legal_country`, `hq_country`, `entity_status`, `watchlist` |
| `SanctionedEntity` | `id` (OpenSanctions) | OpenSanctions | `name`, `schema`, `countries`, `sanctions`, `program_ids`, `dataset`, `last_seen` |
| `NewsEvent` | `id` (GDELT GKG record id) | GDELT | `published_at`, `title`, `url`, `domain`, `event_types`, `tone`, `locations` |

`Company.risk_score` ∈ [0, 1] = `own_risk` + `inherited_risk`; `Company.risk_drivers` (JSON)
lists the signals and link paths that produced it (see [risk.md](risk.md)).

`Company.watchlist` is true for companies SupplyGraph monitors: the seeds, the SEC filers and
both ends of every asserted `SUPPLIES` link (`POSSIBLE_SUPPLIER` leads don't count). Only
watchlist companies are matched against news. `NewsEvent.copies` counts syndicated copies
merged into the event.

`NewsEvent.event_types` ⊆ `NATURAL_DISASTER`, `LABOR_STRIKE`, `SHORTAGE`, `SANCTIONS`,
`CYBER_ATTACK`, `FIRE`, `INDUSTRIAL_ACCIDENT`, `POWER_OUTAGE`, `BLOCKADE`, `BANKRUPTCY`
(mapped from GDELT GKG theme codes; see `ingest/gdelt.py`).

Planned: `Person`, `Facility`, `Product`, `Filing`, `RiskFlag`.

## Link types

| Link | From → To | Properties | Meaning |
|---|---|---|---|
| `OWNED_BY` | `Company` → `Company` | `relationship_type`: `DIRECT` \| `ULTIMATE` | Accounting consolidation parent, as reported to GLEIF |
| `SANCTIONED_AS` | `Company` → `SanctionedEntity` | `match_method`, `confidence`, `evidence` (JSON) | The company and the sanctions record are the same real-world entity |
| `POSSIBLE_SANCTIONS_MATCH` | `Company` → `SanctionedEntity` | `score`, `company_name`, `sanction_name`, `country_relation` | Unresolved candidate awaiting analyst review — **not** an assertion of identity |
| `SUPPLIES` | `Company` (supplier) → `Company` (customer) | `extraction_method`, `evidence`, `supplies`, `mentions`, `form`, `accession`, `filing_date` | The customer's annual report states that it obtains goods or services from the supplier |
| `POSSIBLE_SUPPLIER` | `Company` (supplier) → `Company` (customer) | same as `SUPPLIES` | A low-precision extractor found a supply cue near the name — a lead for review, **not** an assertion |
| `MENTIONED_IN` | `Company` → `NewsEvent` | `matched_name` | A news article tagged with a disruption theme names the company. A signal, **not** evidence that the company was affected |

Planned: `OPERATES_AT`.

### `SUPPLIES` vs `POSSIBLE_SUPPLIER`

The link type follows the extraction method, because the methods' measured precision differs
(see [extraction.md](extraction.md)):

| `extraction_method` | Link | Why |
|---|---|---|
| `LLM` | `SUPPLIES` | Claude classifies every company in the passage by its relationship to the filer; the evidence sentence must occur verbatim in the filing |
| `RULES` | `POSSIBLE_SUPPLIER` | Supply-cue baseline; 0.56 precision on the gold set (it cannot tell suppliers from competitors, customers or JV partners) |

Every supplier link carries the verbatim `evidence` sentence and the filing `accession`, so
any edge can be audited against the source document.

### `SANCTIONED_AS.match_method`

| Method | How | Confidence |
|---|---|---|
| `LEI_IDENTIFIER` | Sanctions record lists the company's LEI | 1.0 |
| `REGISTRATION_NUMBER` | Sanctions record lists the company's local register number (e.g. OGRN), same country, ≥ 6 chars | 1.0 |
| `FUZZY_NAME` | Name + country score ≥ auto-match threshold, multi-token name | < 1.0 |
| `LLM_ADJUDICATED` | Borderline candidate confirmed by Claude; `evidence.reasoning` records why | < 1.0 |
| `ANALYST_CONFIRMED` | A human approved a `confirm_match` action on a review candidate | 1.0 (human-verified) |

A pair with an approved `dismiss_match` action is recorded in `er_candidates` with
`decision = no_match`, `method = ANALYST_DISMISSED`: it leaves the review queue and can never be
auto-matched again.

## Actions

Actions are the only way the agent changes state. The agent can only **propose** an action;
a human approves or rejects it (`supplygraph actions approve|reject`). Every step is an event in
the append-only log `data/actions/events.jsonl` (source of truth, survives warehouse rebuilds);
`build` folds it into the `analyst_actions` table.

| Action | Arguments | Precondition (checked at proposal and approval) | Effect once approved |
|---|---|---|---|
| `flag_supplier` | `lei`, `reason` | company exists | company joins the watchlist (reason `analyst_flag`) at the next `build`; `Company.analyst_flags` |
| `request_review` | `lei`, `reason` | company exists | open item in the analyst queue; `Company.review_requested` |
| `confirm_match` | `lei`, `sanction_id`, `reason` | the pair is in `er_candidates` | `SANCTIONED_AS` with `match_method = ANALYST_CONFIRMED` at the next `resolve` |
| `dismiss_match` | `lei`, `sanction_id`, `reason` | the pair is in `er_candidates` | candidate leaves the review queue (`ANALYST_DISMISSED`) at the next `resolve` |

Every proposal records `proposed_by` (agent model or analyst), the `session_id` and `question`
that led to it, and `evidence`: the refs (`lei:…`, `sanction:…`, `filing:…`, `news:…`) the
agent retrieved that support it — a proposal citing nothing it retrieved is rejected. An
identical open or approved action is never proposed twice. Statuses: `proposed` →
`approved` | `rejected`.

## Agent answers

The Q&A agent answers only from what its read-only tools return, and cites every fact with
the ref of the object it came from: `[lei:<LEI>]`, `[sanction:<id>]`, `[filing:<accession>]`,
`[news:<GDELT id>]`. Text from filings, news and sanctions records is untrusted data, never
instructions.

## Invariants (enforced by `supplygraph check`, Dagster gates and Neo4j constraints)

- `Company.lei` is unique and passes the ISO 7064 MOD 97-10 check.
- No `OWNED_BY` self-loops; both endpoints exist.
- Identifier and analyst-confirmed matches have `confidence = 1.0`; automatic name-based matches
  have `0 < confidence < 1.0`.
- Every `SANCTIONED_AS` carries valid JSON `evidence` and references existing nodes.
- A company resolved by identifier is never additionally fuzzy-matched.
- Supplier links reference existing companies, are never self-loops, carry non-empty
  `evidence`, and their link type matches their `extraction_method`.
- Every company has a risk score in [0, 1]; every non-zero score is explained by drivers.
- Every `NewsEvent` has at least one known event type; `MENTIONED_IN` only points from
  watchlist companies.
- Every analyst action has a known type and status, a reason, and evidence; every decided action
  records who decided and when.
