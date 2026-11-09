# 0007. Explainable rule-based risk propagation; no GNN at this scale

**Status:** Accepted (M5)

## Context
Analysts must defend a risk flag, for example "why is Rosneft Deutschland at 0.81?". The graph
is small (278 companies, 239 ownership links, 30 supplier leads), and there are no labelled
outcomes to train against. The original roadmap said "link prediction with PyTorch Geometric".

## Decision
- **Propagation.** Own-risk signals (sanctions match, review candidate, decayed news) travel
  along the single strongest hop-limited path, with a weight for each link type.
- **Combination.** Signals combine by noisy-OR, which is bounded, monotone and cycle-safe.
- **Explanation.** Every non-zero score stores its drivers: signal, source and path.
- **Routing.** `SIGNAL_RELATIONS` decides which links each signal travels. Sanctions follow
  ownership; disruptions follow supply and child → parent links.
- **No GNN.** Link prediction is benchmarked leave-one-out, and a learned model ships only if
  it beats the popularity baseline. Item-based collaborative filtering scored below random.

## Alternatives considered
- **A GNN or learned scorer.** There are no labels to learn from. It would bring a ~700 MB
  dependency, and nothing would show whether it is right
  (decision record in [risk.md](../risk.md)).

## Consequences
- Weights are judgement calls. `evaluate-risk` measures how sensitive the ranking is to them
  (every weight ×0.5 / ×1.5 keeps ≥ 9 of the top 10).
- **Revisit when** there are several hundred asserted `SUPPLIES` edges.
