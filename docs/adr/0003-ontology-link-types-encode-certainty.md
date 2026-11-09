# 0003. The ontology is the contract; link types encode certainty

**Status:** Accepted (M1, extended in M3, M4 and M6)

## Context
Sources disagree in reliability. An LEI listed on a sanctions record is a fact. A fuzzy name
match is a probability. A regex hit near "supplier" is a lead that is right about half the time.
If these all become the same kind of edge, every consumer (risk, agent, UI, an analyst in Neo4j
Browser) has to remember to check a confidence property, and some won't.

## Decision
`docs/ontology.md` defines every object, link and action type. It is updated *before* any
change to them. Link types themselves separate assertions from leads:

| Assertion | Lead |
|---|---|
| `SANCTIONED_AS` (identifier, auto fuzzy ≥ 0.97, LLM-adjudicated, analyst-confirmed) | `POSSIBLE_SANCTIONS_MATCH` (review queue) |
| `SUPPLIES` (LLM extraction, verbatim evidence) | `POSSIBLE_SUPPLIER` (rule extraction, 0.56 precision) |

The link type follows the *method*, whose precision is measured, and a quality check enforces
it.

## Consequences
- A Cypher query or a risk rule that ignores leads only has to name the asserted type.
- The risk engine gives leads lower transmission weights (0.25 vs 0.5 for supply), and the agent
  and UI label them "unverified".
- Promoting a lead to an assertion is an explicit act: re-extraction with Claude, or an approved
  `confirm_match` (0009). A confidence value drifting over a threshold never does it.
