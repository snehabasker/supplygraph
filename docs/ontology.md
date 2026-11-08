# SupplyGraph ontology (v0.2)

The ontology is the contract between data pipelines, the graph, the agent and the UI.
Every source is mapped *into* these types; nothing downstream reads raw source fields.

## Object types

| Type | Key | Source | Key properties |
|---|---|---|---|
| `Company` | `lei` (ISO 17442) | GLEIF | `legal_name`, `other_names`, `registered_as`, `jurisdiction`, `legal_country`, `hq_country`, `entity_status` |
| `SanctionedEntity` | `id` (OpenSanctions) | OpenSanctions | `name`, `schema`, `countries`, `sanctions`, `program_ids`, `dataset`, `last_seen` |

Planned: `Person`, `Facility`, `Product`, `Event` (news/disruption), `Filing`, `RiskFlag`.

## Link types

| Link | From → To | Properties | Meaning |
|---|---|---|---|
| `OWNED_BY` | `Company` → `Company` | `relationship_type`: `DIRECT` \| `ULTIMATE` | Accounting consolidation parent, as reported to GLEIF |
| `SANCTIONED_AS` | `Company` → `SanctionedEntity` | `match_method`, `confidence`, `evidence` (JSON) | The company and the sanctions record are the same real-world entity |
| `POSSIBLE_SANCTIONS_MATCH` | `Company` → `SanctionedEntity` | `score`, `company_name`, `sanction_name`, `country_relation` | Unresolved candidate awaiting analyst review — **not** an assertion of identity |

Planned: `SUPPLIES` (from 10-K extraction), `AFFECTED_BY` (news events), `OPERATES_AT`.

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
