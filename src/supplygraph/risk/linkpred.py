"""Supplier recommendation ("which suppliers might this company also depend on?") as link
prediction on the bipartite supplier -> customer graph, evaluated leave-one-out.

For every customer with at least two known suppliers, hide one link, rank all other known
suppliers for that customer, and record where the hidden one lands. Methods:

* random      — expected rank of a uniformly random ordering (computed exactly);
* popularity  — suppliers with more customers first (TSMC first, for everyone);
* cooccurrence — item-based collaborative filtering: suppliers that share customers with this
                 customer's other suppliers ("companies that use Samsung also use SK hynix");
* hybrid      — cooccurrence, ties broken by popularity.

With ~30 links this is a feasibility study, not a model: the report gives the number of
queries, and docs/risk.md says what the numbers can and cannot support. A graph neural network
was deliberately not used at this scale (see the ADR in docs/risk.md).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import duckdb

Links = list[tuple[str, str]]  # (supplier, customer)
Scorer = Callable[[str, Links], dict[str, float]]


def popularity(customer: str, links: Links) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    for supplier, _ in links:
        counts[supplier] += 1
    return dict(counts)


def cooccurrence(customer: str, links: Links) -> dict[str, float]:
    customers_of: dict[str, set[str]] = defaultdict(set)
    for supplier, cust in links:
        customers_of[supplier].add(cust)
    mine = {s for s, c in links if c == customer}
    scores = {}
    for cand, cand_customers in customers_of.items():
        total = 0.0
        for s in mine:
            shared = len(cand_customers & customers_of[s] - {customer})
            if shared:
                total += shared / math.sqrt(len(cand_customers) * len(customers_of[s]))
        scores[cand] = total
    return scores


def hybrid(customer: str, links: Links) -> dict[str, float]:
    co, pop = cooccurrence(customer, links), popularity(customer, links)
    top = max(pop.values(), default=1.0)
    return {s: co.get(s, 0.0) + 1e-3 * pop.get(s, 0.0) / top for s in pop}


SCORERS: dict[str, Scorer] = {"popularity": popularity, "cooccurrence": cooccurrence,
                              "hybrid": hybrid}


def rank_of(target: str, scores: dict[str, float], candidates: list[str]) -> float:
    """1-based rank of `target`; ties share the average rank (no lucky tie-breaking)."""
    t = scores.get(target, 0.0)
    better = sum(1 for c in candidates if scores.get(c, 0.0) > t)
    tied = sum(1 for c in candidates if scores.get(c, 0.0) == t)
    return better + (tied + 1) / 2


def leave_one_out(links: Links, scorers: dict[str, Scorer] | None = None,
                  ks: tuple[int, ...] = (1, 3)) -> dict:
    scorers = scorers or SCORERS
    links = sorted(set(links))
    suppliers_of: dict[str, set[str]] = defaultdict(set)
    for s, c in links:
        suppliers_of[c].add(s)
    all_suppliers = sorted({s for s, _ in links})
    queries = []
    for customer in sorted(suppliers_of):
        if len(suppliers_of[customer]) < 2:
            continue
        for hidden in sorted(suppliers_of[customer]):
            train = [(s, c) for s, c in links if (s, c) != (hidden, customer)]
            known = {s for s, c in train if c == customer}
            candidates = [s for s in all_suppliers if s not in known]
            queries.append((customer, hidden, train, candidates))

    report: dict = {"links": len(links), "customers": len(suppliers_of),
                    "queries": len(queries), "candidates_per_query":
                    round(sum(len(q[3]) for q in queries) / len(queries), 1) if queries else 0,
                    "methods": {}}
    if not queries:
        return report
    # Uniformly random ordering: rank is uniform on 1..n, so these are exact expectations.
    rand = {"mrr": sum(sum(1 / r for r in range(1, len(q[3]) + 1)) / len(q[3])
                       for q in queries) / len(queries)}
    for k in ks:
        rand[f"hit@{k}"] = sum(min(k, len(q[3])) / len(q[3]) for q in queries) / len(queries)
    report["methods"]["random"] = {m: round(v, 3) for m, v in rand.items()}
    for name, scorer in scorers.items():
        ranks = [rank_of(hidden, scorer(customer, train), candidates)
                 for customer, hidden, train, candidates in queries]
        m = {"mrr": sum(1 / r for r in ranks) / len(ranks)}
        for k in ks:
            m[f"hit@{k}"] = sum(r <= k for r in ranks) / len(ranks)
        report["methods"][name] = {key: round(v, 3) for key, v in m.items()}
    return report


def load_links(con: duckdb.DuckDBPyConnection) -> Links:
    return [(s, c) for s, c in con.execute(
        "SELECT DISTINCT supplier_lei, customer_lei FROM supply_links").fetchall()]


def recommend(con: duckdb.DuckDBPyConnection, customer: str, top: int = 5,
              method: str = "popularity") -> list[tuple[str, float]]:
    """Default = the best method in the leave-one-out benchmark on the live graph (popularity
    beat co-occurrence, which was below random at 30 links; see docs/risk.md)."""
    links = load_links(con)
    known = {s for s, c in links if c == customer}
    scores = SCORERS[method](customer, links)
    ranked = sorted(((s, v) for s, v in scores.items() if s not in known and s != customer),
                    key=lambda kv: -kv[1])
    return ranked[:top]
