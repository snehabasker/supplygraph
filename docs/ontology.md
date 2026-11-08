# SupplyGraph ontology (v0.3)

The ontology is the contract between data pipelines, the graph, the agent and the UI.
Every source is mapped *into* these types; nothing downstream reads raw source fields.

## Object types

| Type | Key | Source | Key properties |
|---|---|---|---|
| `Company` | `lei` (ISO 17442) | GLEIF | `legal_name`, `other_names`, `registered_as`, `jurisdiction`, `legal_country`, `hq_country`, `entity_status`, `watchlist` |
| `SanctionedEntity` | `id` (OpenSanctions) | OpenSanctions | `name`, `schema`, `countries`, `sanctions`, `program_ids`, `dataset`, `last_seen` |
| `NewsEvent` | `id` (GDELT GKG record id) | GDELT | `published_at`, `title`, `url`, `domain`, `event_types`, `tone`, `locations` |

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

## Actions (Milestone 6)

Actions are the only way the agent changes state, and every action is audited:
`flag_supplier(lei, reason)`, `request_review(lei)`, `confirm_match(lei, sanction_id)`,
`dismiss_match(lei, sanction_id, reason)`.

## Invariants (enforced by `supplygraph check`, Dagster gates and Neo4j constraints)

- `Company.lei` is unique and passes the ISO 7064 MOD 97-10 check.
- No `OWNED_BY` self-loops; both endpoints exist.
- Identifier matches have `confidence = 1.0`; name-based matches have `0 < confidence < 1.0`.
- Every `SANCTIONED_AS` carries valid JSON `evidence` and references existing nodes.
- A company resolved by identifier is never additionally fuzzy-matched.
- Supplier links reference existing companies, are never self-loops, carry non-empty
  `evidence`, and their link type matches their `extraction_method`.
- Every `NewsEvent` has at least one known event type; `MENTIONED_IN` only points from
  watchlist companies.
