# Risk scoring, scenarios and supplier recommendation

**Question it answers:** *How exposed is this company — to sanctions through its owners, and to
disruptions through its suppliers — and why?* Every score comes with the signal and the path
that produced it, because a risk score an analyst cannot explain is a score they cannot act on.

## Model

**Own risk** comes from signals attached to the company:

| Signal | Value | Why this strength |
|---|---|---|
| `SANCTIONED` | 1.0 × match confidence | Identifier matches are certain (1.0); fuzzy / LLM matches carry their confidence |
| `POSSIBLE_SANCTION` | 0.3 × candidate score | A review-queue lead, not a finding |
| `NEWS` | 0.05 per article, half-life 3 days, capped at 0.2 | `MENTIONED_IN` links are ~1-in-3 relevant (docs/extraction.md) |
| `SCENARIO` | user-chosen severity | "What if X is disrupted?" |

**Inherited risk** flows along links, each with a transmission factor:

| Link (direction risk flows) | Factor | Carries |
|---|---:|---|
| parent → subsidiary, direct (`OWNED_BY:DIRECT`) | 0.8 | sanctions |
| parent → subsidiary, ultimate | 0.6 | sanctions |
| subsidiary → parent (`PARENT_OF`) | 0.3 | sanctions, disruptions |
| supplier → customer (`SUPPLIES`) | 0.5 | sanctions, disruptions |
| supplier → customer (`POSSIBLE_SUPPLIER`) | 0.25 | sanctions, disruptions |

*Which* links a signal may travel depends on its kind. Sanctions follow legal control — a
subsidiary of a sanctioned parent is itself restricted (the idea behind OFAC's 50 % rule) — as
well as supply. Operational disruption follows physical dependency: a fab fire hurts the fab's
customers and its parent's results, but news about Walmart does not disrupt Flipkart. (The first
version let news flow parent → subsidiary; on live data every Walmart and NVIDIA subsidiary
inherited "risk" from market-wrap articles.) `POSSIBLE_SUPPLIER` gets half the `SUPPLIES` factor,
roughly in line with the rule extractor's 0.56 precision.

**Combination.** Each signal travels along its single strongest path (product of factors,
≤ 4 hops, contributions < 0.01 dropped); the contributions of different signals combine as a
noisy-OR, `1 − Π(1 − cᵢ)`. Properties, all tested (`tests/test_risk.py`, including a
randomized monotonicity test):

- scores are in [0, 1] and `own + inherited = total`;
- adding a signal or a link never lowers any score;
- cycles cannot amplify risk; a company with no path to any signal scores exactly 0;
- every non-zero score stores its top drivers (signal, source, value, path, link types) —
  enforced by the `risk_scores_explained` quality check, so an unexplained score blocks the
  graph load.

## Results on the live graph (278 companies)

24 companies carry risk; 17 of them only through links.

| Company | Risk | Own | Top driver |
|---|---:|---:|---|
| PJSC Rosneft Oil Company | 1.00 | 1.00 | LEI listed on a sanctions record |
| RN-Capital LLC | 0.86 | 0.29 | parent Rosneft (+ its own review-queue candidate) |
| Rosneft Deutschland GmbH | 0.81 | 0 | parent Rosneft, 1 hop `OWNED_BY:DIRECT` |
| ASML Hong Kong Limited | 0.28 | 0.28 | sanctions review-queue candidate |
| NVIDIA Corporation | 0.20 | 0.20 | disruption news (capped) |

**Scenario: TSMC disrupted** (`supplygraph scenario TSMC`): AMD, Qualcomm, Intel, Broadcom and
NVIDIA rise by 0.20–0.25 via `POSSIBLE_SUPPLIER`; so does ASML — through the *false* lead
"TSMC supplies ASML" (TSMC is ASML's customer). The path makes the error visible, which is the
point of keeping leads separate from assertions.

## Evaluation

There are no ground-truth risk labels, so the engine is evaluated on what *can* be checked:

1. **Properties** (above) and **known cases**: the Rosneft subsidiaries are flagged through
   ownership; unconnected companies score 0.
2. **Sensitivity to the weights** (`supplygraph evaluate-risk`): scaling each of the 8 weights
   by ×0.5 and ×1.5 keeps **≥ 9 of the top 10** at-risk companies and **Kendall τ ≥ 0.85** in
   every variation. Caveat: with few signals, the ranking is dominated by the Rosneft cluster,
   which any sensible weighting puts first.

## Supplier recommendation (link prediction)

Task: hide one known supplier link, rank all other suppliers for that customer, see where the
hidden one lands (leave-one-out over every customer with ≥ 2 suppliers).

| Method (29 queries, ~19 candidates each) | MRR | hit@1 | hit@3 |
|---|---:|---:|---:|
| random (exact expectation) | 0.191 | 0.054 | 0.163 |
| **popularity** — suppliers with most customers first | **0.328** | **0.241** | **0.310** |
| co-occurrence (item-based collaborative filtering) | 0.167 | 0.034 | 0.103 |
| hybrid (co-occurrence, popularity tie-break) | 0.219 | 0.103 | 0.172 |

Popularity wins; collaborative filtering is *below random*. With 30 links (and those are
0.56-precision leads), almost no two customers share enough suppliers for "companies like you
also use…" to carry signal — the hub (TSMC) is the only pattern. `recommend()` therefore
defaults to popularity. 29 queries is a small sample: the ranking popularity > random is
clear, differences between the weaker methods are not.

## Decision record: no graph neural network (yet)

**Context.** The roadmap originally said "link prediction with PyTorch Geometric".
**Decision.** Not at this scale: 278 nodes, 30 supplier edges, all of them noisy leads.
**Why.** A GNN learns from many repeated structural patterns; here even item-based
collaborative filtering — a far simpler learner — loses to "recommend the most popular
supplier". A GNN would add a ~700 MB dependency and an unfalsifiable result.
**Revisit when** there are several hundred asserted `SUPPLIES` edges (Claude extraction over
many more filers) — and then it must beat the popularity baseline in this same benchmark.
