# SupplyGraph ontology (v0.1)

The ontology is the contract between data pipelines, the graph, the agent and the UI.
Every source is mapped *into* these types; nothing downstream reads raw source fields.

## Object types

| Type | Key | Source (v0.1) | Key properties |
|---|---|---|---|
| `Company` | `lei` (ISO 17442) | GLEIF | `legal_name`, `jurisdiction`, `legal_country`, `hq_country`, `entity_status`, `registration_status` |
| `SanctionedEntity` | `id` (OpenSanctions) | OpenSanctions | `name`, `schema`, `countries`, `sanctions`, `program_ids`, `dataset`, `last_seen` |

Planned: `Person`, `Facility`, `Product`, `Event` (news/disruption), `Filing`, `RiskFlag`.

## Link types

| Link | From → To | Properties | Meaning |
|---|---|---|---|
| `OWNED_BY` | `Company` → `Company` | `relationship_type`: `DIRECT` \| `ULTIMATE` | Accounting consolidation parent, as reported to GLEIF |
| `SANCTIONED_AS` | `Company` → `SanctionedEntity` | `match_method`, `confidence` | The company and the sanctions record are the same real-world entity |

Planned: `SUPPLIES` (from 10-K extraction), `AFFECTED_BY` (news events), `OPERATES_AT`.

## Actions (Milestone 6)

Actions are the only way the agent changes state, and every action is audited:
`flag_supplier(lei, reason)`, `request_review(lei)`, `dismiss_match(lei, sanction_id, reason)`.

## Invariants (enforced by `supplygraph check` and Neo4j constraints)

- `Company.lei` is unique and passes the ISO 7064 MOD 97-10 check.
- No `OWNED_BY` self-loops; both endpoints exist.
- `SANCTIONED_AS` with `match_method = LEI_IDENTIFIER` has `confidence = 1.0`;
  fuzzy matches (Milestone 3) must carry `confidence < 1.0` and evidence.
