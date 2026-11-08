"""Explainable risk propagation.

A company's risk has two parts:

* **own risk** — signals attached to the company itself: an identifier or name match to a
  sanctions record, a candidate in the sanctions review queue, recent disruption news;
* **inherited risk** — the same signals arriving over graph links: a subsidiary inherits from
  its sanctioned parent (the logic behind OFAC's 50 % rule), a customer inherits from a
  disrupted supplier.

Every signal travels along its single strongest path: its value times the product of the
transmission factors on the path, over at most `max_hops` links. Different signals reaching a
company combine as a noisy-OR, ``1 - prod(1 - c_i)``. Consequences, each covered by a test:

* scores stay in [0, 1]; adding a signal or a link never lowers any score;
* cycles cannot amplify risk (a path is only as strong as its weakest factor);
* every non-zero score is explained by the stored source signal and path that produced it.

The weights are judgement calls, documented in docs/risk.md; `sensitivity()` measures how
much the ranking depends on them.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime

import duckdb

from supplygraph.warehouse import table_exists

log = logging.getLogger(__name__)

SANCTIONED = "SANCTIONED"
POSSIBLE_SANCTION = "POSSIBLE_SANCTION"
NEWS = "NEWS"
SCENARIO = "SCENARIO"


@dataclass(frozen=True)
class RiskConfig:
    # Own-risk signal strengths.
    sanctioned: float = 1.0  # x match confidence
    possible_sanction: float = 0.3  # x candidate score: a lead, not a finding
    news_event: float = 0.05  # per linked event; news links are ~1-in-3 relevant
    news_cap: float = 0.2
    news_half_life_days: float = 3.0
    # Transmission factors: the share of a company's risk that reaches its neighbour.
    parent_to_child_direct: float = 0.8
    parent_to_child_ultimate: float = 0.6
    child_to_parent: float = 0.3
    supplier_to_customer: float = 0.5  # SUPPLIES
    possible_supplier_to_customer: float = 0.25  # POSSIBLE_SUPPLIER: ~0.56 precision
    max_hops: int = 4
    min_contribution: float = 0.01

    def edge_factor(self, relation: str) -> float:
        return {
            "OWNED_BY:DIRECT": self.parent_to_child_direct,
            "OWNED_BY:ULTIMATE": self.parent_to_child_ultimate,
            "PARENT_OF": self.child_to_parent,
            "SUPPLIES": self.supplier_to_customer,
            "POSSIBLE_SUPPLIER": self.possible_supplier_to_customer,
        }[relation]


SANCTION_RELATIONS = frozenset({"OWNED_BY:DIRECT", "OWNED_BY:ULTIMATE", "PARENT_OF",
                                "SUPPLIES", "POSSIBLE_SUPPLIER"})
DISRUPTION_RELATIONS = frozenset({"PARENT_OF", "SUPPLIES", "POSSIBLE_SUPPLIER"})

# Which links each kind of risk travels. Sanctions follow legal control (a subsidiary of a
# sanctioned parent is itself restricted) as well as supply. Operational disruption follows
# physical dependency: a fab fire hurts the fab's customers and its parent's results, but news
# about Walmart does not disrupt Flipkart — on live data, without this rule, every subsidiary
# of a company in the news inherited its "risk".
SIGNAL_RELATIONS = {
    SANCTIONED: SANCTION_RELATIONS,
    POSSIBLE_SANCTION: SANCTION_RELATIONS,
    NEWS: DISRUPTION_RELATIONS,
    SCENARIO: DISRUPTION_RELATIONS,
}

TUNABLE = ("sanctioned", "possible_sanction", "news_event", "parent_to_child_direct",
           "parent_to_child_ultimate", "child_to_parent", "supplier_to_customer",
           "possible_supplier_to_customer")


@dataclass(frozen=True)
class Signal:
    lei: str
    kind: str
    value: float
    detail: str


@dataclass(frozen=True)
class Edge:
    """Risk flows from `src` to `dst`."""
    src: str
    dst: str
    relation: str


@dataclass(frozen=True)
class Contribution:
    target: str
    signal: Signal
    value: float
    path: tuple[str, ...]  # source ... target
    relations: tuple[str, ...]  # one per hop

    def as_dict(self, names: dict[str, str] | None = None) -> dict:
        names = names or {}
        return {
            "signal": self.signal.kind, "detail": self.signal.detail,
            "source": self.signal.lei, "source_name": names.get(self.signal.lei),
            "value": round(self.value, 4), "hops": len(self.relations),
            "path": [names.get(lei, lei) for lei in self.path],
            "relations": list(self.relations),
        }


@dataclass
class RiskResult:
    scores: dict[str, float]
    own: dict[str, float]
    contributions: dict[str, list[Contribution]] = field(default_factory=dict)

    def drivers(self, lei: str, top: int = 3) -> list[Contribution]:
        return sorted(self.contributions.get(lei, []), key=lambda c: -c.value)[:top]


def noisy_or(values: list[float]) -> float:
    return 1.0 - math.prod(1.0 - min(max(v, 0.0), 1.0) for v in values)


def best_paths(
    signal: Signal, out_edges: dict[str, list[Edge]], cfg: RiskConfig
) -> dict[str, tuple[float, tuple[str, ...], tuple[str, ...]]]:
    """Strongest hop-limited path from the signal's company to every company it reaches:
    Bellman-Ford over max-product, `max_hops` rounds. Exact under the hop limit."""
    allowed = SIGNAL_RELATIONS.get(signal.kind, SANCTION_RELATIONS)
    best = {signal.lei: (signal.value, (signal.lei,), ())}
    frontier = dict(best)
    for _ in range(cfg.max_hops):
        nxt: dict[str, tuple[float, tuple[str, ...], tuple[str, ...]]] = {}
        for node, (value, path, rels) in frontier.items():
            for e in out_edges.get(node, []):
                if e.relation not in allowed or e.dst in path:
                    continue  # wrong kind of link for this risk, or a cycle
                v = value * cfg.edge_factor(e.relation)
                if v < cfg.min_contribution:
                    continue
                if v > best.get(e.dst, (0.0,))[0] and v > nxt.get(e.dst, (0.0,))[0]:
                    nxt[e.dst] = (v, (*path, e.dst), (*rels, e.relation))
        for node, item in nxt.items():
            if item[0] > best.get(node, (0.0,))[0]:
                best[node] = item
        frontier = nxt
        if not frontier:
            break
    return best


def propagate(signals: list[Signal], edges: list[Edge], cfg: RiskConfig | None = None,
              companies: list[str] | None = None) -> RiskResult:
    cfg = cfg or RiskConfig()
    out_edges: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        if e.src != e.dst:
            out_edges[e.src].append(e)
    contributions: dict[str, list[Contribution]] = defaultdict(list)
    for s in signals:
        if s.value <= 0:
            continue
        for target, (value, path, rels) in best_paths(s, out_edges, cfg).items():
            contributions[target].append(Contribution(target, s, value, path, rels))
    leis = set(companies or []) | set(contributions)
    scores = {lei: noisy_or([c.value for c in contributions.get(lei, [])]) for lei in leis}
    own = {lei: noisy_or([c.value for c in contributions.get(lei, []) if not c.relations])
           for lei in leis}
    return RiskResult(scores, own, dict(contributions))


# --- loading from the warehouse ----------------------------------------------------------

def load_edges(con: duckdb.DuckDBPyConnection) -> list[Edge]:
    edges = []
    for child, parent, rel_type in con.execute(
        "SELECT child_lei, parent_lei, relationship_type FROM ownership"
    ).fetchall():
        edges.append(Edge(parent, child, f"OWNED_BY:{rel_type}"))  # parent's risk -> child
        edges.append(Edge(child, parent, "PARENT_OF"))  # weaker: child's risk -> parent
    if table_exists(con, "supply_links"):
        for supplier, customer, link_type in con.execute(
            "SELECT supplier_lei, customer_lei, link_type FROM supply_links"
        ).fetchall():
            edges.append(Edge(supplier, customer, link_type))
    return edges


def _decay(published_at: datetime | None, now: datetime, half_life_days: float) -> float:
    if published_at is None:
        return 0.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    age_days = max(0.0, (now - published_at).total_seconds() / 86400)
    return 0.5 ** (age_days / half_life_days)


def load_signals(con: duckdb.DuckDBPyConnection, cfg: RiskConfig,
                 now: datetime | None = None) -> list[Signal]:
    now = now or datetime.now(UTC)
    signals = []
    for lei, sid, method, confidence in con.execute(
        "SELECT lei, sanction_id, match_method, confidence FROM company_sanction_matches"
    ).fetchall():
        signals.append(Signal(lei, SANCTIONED, cfg.sanctioned * float(confidence),
                              f"{method} match to {sid}"))
    if table_exists(con, "er_candidates"):
        for lei, sid, score in con.execute(
            "SELECT lei, sanction_id, score FROM er_candidates WHERE decision = 'review'"
        ).fetchall():
            signals.append(Signal(lei, POSSIBLE_SANCTION, cfg.possible_sanction * float(score),
                                  f"review-queue candidate {sid} (score {score:.2f})"))
    if table_exists(con, "company_news"):
        per_company: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for lei, title, types, published_at in con.execute("""
            SELECT n.lei, e.title, e.event_types, e.published_at
            FROM company_news n JOIN news_events e ON e.id = n.event_id
        """).fetchall():
            weight = cfg.news_event * _decay(published_at, now, cfg.news_half_life_days)
            per_company[lei].append((weight, f"{'/'.join(types)}: {title}"))
        for lei, items in per_company.items():
            total = min(cfg.news_cap, sum(w for w, _ in items))
            if total > 0:
                top = max(items)[1]
                signals.append(Signal(lei, NEWS, total,
                                      f"{len(items)} disruption article(s), e.g. {top[:120]}"))
    return signals


def company_names(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    return dict(con.execute("SELECT lei, legal_name FROM companies").fetchall())


def score(con: duckdb.DuckDBPyConnection, cfg: RiskConfig | None = None,
          extra_signals: list[Signal] | None = None, now: datetime | None = None) -> RiskResult:
    cfg = cfg or RiskConfig()
    leis = [lei for (lei,) in con.execute("SELECT lei FROM companies").fetchall()]
    signals = load_signals(con, cfg, now) + list(extra_signals or [])
    return propagate(signals, load_edges(con), cfg, leis)


def write_scores(con: duckdb.DuckDBPyConnection, result: RiskResult, cfg: RiskConfig) -> dict:
    names = company_names(con)
    con.execute("""
        CREATE OR REPLACE TABLE company_risk (
            lei VARCHAR, risk_score DOUBLE, own_risk DOUBLE, inherited_risk DOUBLE,
            drivers VARCHAR
        )
    """)
    rows = []
    for lei, s in result.scores.items():
        own = result.own.get(lei, 0.0)
        drivers = [c.as_dict(names) for c in result.drivers(lei)]
        rows.append([lei, round(s, 6), round(own, 6), round(max(0.0, s - own), 6),
                     json.dumps(drivers, ensure_ascii=False)])
    if rows:
        con.executemany("INSERT INTO company_risk VALUES (?, ?, ?, ?, ?)", rows)
    con.execute("CREATE OR REPLACE TABLE risk_config AS SELECT ? AS config, now() AS scored_at",
                [json.dumps(asdict(cfg))])
    at_risk = sum(1 for _, s, *_ in rows if s > 0)
    return {"companies_scored": len(rows), "companies_at_risk": at_risk,
            "inherited_only": sum(1 for r in rows if r[1] > 0 and r[2] == 0)}


# --- scenarios and sensitivity -----------------------------------------------------------

def scenario(con: duckdb.DuckDBPyConnection, lei: str, severity: float = 1.0,
             cfg: RiskConfig | None = None, now: datetime | None = None) -> list[dict]:
    """What changes if `lei` is disrupted? Companies whose score rises, with the path."""
    cfg = cfg or RiskConfig()
    base = score(con, cfg, now=now)
    shock = Signal(lei, SCENARIO, severity, "scenario: disruption")
    shocked = score(con, cfg, [shock], now=now)
    names = company_names(con)
    out = []
    for target, s in shocked.scores.items():
        delta = s - base.scores.get(target, 0.0)
        if target == lei or delta < 1e-9:
            continue
        via = next(c for c in shocked.contributions[target] if c.signal == shock)
        out.append({"lei": target, "name": names.get(target), "before": round(base.scores
                    .get(target, 0.0), 4), "after": round(s, 4), "delta": round(delta, 4),
                    "path": [names.get(x, x) for x in via.path], "relations": list(via.relations)})
    return sorted(out, key=lambda r: -r["delta"])


def _ranking(result: RiskResult) -> list[str]:
    return [lei for lei, s in sorted(result.scores.items(), key=lambda kv: (-kv[1], kv[0]))
            if s > 0]


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall's tau-a over the items both rankings contain."""
    common = [x for x in a if x in set(b)]
    pos = {x: i for i, x in enumerate(b)}
    n = len(common)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos[common[i]] < pos[common[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def sensitivity(con: duckdb.DuckDBPyConnection, cfg: RiskConfig | None = None,
                factors: tuple[float, ...] = (0.5, 1.5), top_k: int = 10,
                now: datetime | None = None) -> dict:
    """Scale each weight by each factor and compare the ranking of at-risk companies with the
    baseline: top-k overlap and Kendall's tau."""
    cfg = cfg or RiskConfig()
    edges, leis = load_edges(con), [r[0] for r in con.execute("SELECT lei FROM companies")
                                    .fetchall()]

    def run(c: RiskConfig) -> RiskResult:
        return propagate(load_signals(con, c, now), edges, c, leis)

    baseline = _ranking(run(cfg))
    rows = []
    for name in TUNABLE:
        for f in factors:
            varied = replace(cfg, **{name: min(1.0, getattr(cfg, name) * f)})
            ranked = _ranking(run(varied))
            k = min(top_k, len(baseline))
            overlap = len(set(baseline[:k]) & set(ranked[:k])) / k if k else 1.0
            rows.append({"weight": name, "factor": f, "top_k_overlap": round(overlap, 3),
                         "kendall_tau": round(kendall_tau(baseline, ranked), 3),
                         "at_risk": len(ranked)})
    return {"baseline_at_risk": len(baseline), "top_k": top_k, "variations": rows,
            "min_top_k_overlap": min((r["top_k_overlap"] for r in rows), default=1.0),
            "min_kendall_tau": min((r["kendall_tau"] for r in rows), default=1.0)}
