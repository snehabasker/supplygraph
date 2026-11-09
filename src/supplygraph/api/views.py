"""Read models for the explorer UI: dashboard summary and Cytoscape-ready subgraphs.

Company-level detail reuses the agent's `GraphTools`, so the UI and the agent see the graph
through the same queries and the same semantics.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from itertools import pairwise

import duckdb

from supplygraph.agent.actions import PROPOSED, ActionStore
from supplygraph.agent.tools import GraphTools
from supplygraph.warehouse import table_exists

# Upper bounds (exclusive of 0, inclusive above) of the risk histogram buckets.
RISK_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
MAX_NODES = 150


def _count(con: duckdb.DuckDBPyConnection, table: str, where: str = "") -> int:
    if not table_exists(con, table):
        return 0
    row = con.execute(f"SELECT count(*) FROM {table} {where}").fetchone()
    return int(row[0]) if row else 0


def summary(con: duckdb.DuckDBPyConnection, store: ActionStore | None = None) -> dict:
    scores = [s for (s,) in con.execute("SELECT risk_score FROM company_risk").fetchall()] \
        if table_exists(con, "company_risk") else []
    buckets, low = [], 0.0
    for high in RISK_BUCKETS:
        buckets.append({"from": low, "to": high,
                        "companies": sum(1 for s in scores if low < s <= high)})
        low = high
    supply = dict(con.execute("SELECT link_type, count(*) FROM supply_links GROUP BY 1")
                  .fetchall()) if table_exists(con, "supply_links") else {}
    scored_at = con.execute("SELECT scored_at FROM risk_config").fetchone() \
        if table_exists(con, "risk_config") else None
    actions = store.actions() if store is not None else []
    return {
        "companies": _count(con, "companies"),
        "at_risk": sum(1 for s in scores if s > 0),
        "sanctioned": con.execute("SELECT count(DISTINCT lei) FROM company_sanction_matches")
        .fetchone()[0] if table_exists(con, "company_sanction_matches") else 0,
        "review_queue": _count(con, "er_candidates", "WHERE decision = 'review'"),
        "watchlist": _count(con, "watchlist"),
        "ownership_links": _count(con, "ownership"),
        "supply_links": {"SUPPLIES": supply.get("SUPPLIES", 0),
                         "POSSIBLE_SUPPLIER": supply.get("POSSIBLE_SUPPLIER", 0)},
        "news_links": _count(con, "company_news"),
        "proposed_actions": sum(1 for a in actions if a.status == PROPOSED),
        "risk_histogram": buckets,
        "zero_risk": sum(1 for s in scores if s <= 0),
        "scored_at": scored_at[0].isoformat() if scored_at and scored_at[0] else None,
    }


def _edges(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Every ownership and supplier link, in the direction the ontology defines."""
    edges = [{"source": c, "target": p, "type": "OWNED_BY", "relationship_type": t}
             for c, p, t in con.execute(
                 "SELECT child_lei, parent_lei, relationship_type FROM ownership").fetchall()]
    if table_exists(con, "supply_links"):
        edges += [{"source": s, "target": c, "type": lt, "evidence": ev, "accession": acc,
                   "supplies": sup}
                  for s, c, lt, ev, acc, sup in con.execute(
                      "SELECT supplier_lei, customer_lei, link_type, evidence, accession, "
                      "supplies FROM supply_links").fetchall()]
    for e in edges:
        e["id"] = f"{e['type']}:{e['source']}->{e['target']}"
    return edges


def _nodes(con: duckdb.DuckDBPyConnection, leis: list[str]) -> list[dict]:
    flags = {}
    if table_exists(con, "watchlist"):
        flags["watchlist"] = {r[0] for r in con.execute("SELECT lei FROM watchlist").fetchall()}
    flags["sanctioned"] = {r[0] for r in con.execute(
        "SELECT lei FROM company_sanction_matches").fetchall()}
    if table_exists(con, "er_candidates"):
        flags["review"] = {r[0] for r in con.execute(
            "SELECT lei FROM er_candidates WHERE decision = 'review'").fetchall()}
    risk_join = ("LEFT JOIN company_risk r USING (lei)" if table_exists(con, "company_risk")
                 else "")
    risk_cols = ("coalesce(r.risk_score, 0), coalesce(r.own_risk, 0)" if risk_join
                 else "0.0, 0.0")
    rows = con.execute(
        f"SELECT c.lei, c.legal_name, c.legal_country, {risk_cols} FROM companies c "
        f"{risk_join} WHERE c.lei IN (SELECT unnest(?))", [leis]).fetchall()
    return [{"id": lei, "name": name, "country": country, "risk": risk, "own_risk": own,
             **{k: lei in v for k, v in flags.items()}}
            for lei, name, country, risk, own in sorted(rows)]


def neighborhood(con: duckdb.DuckDBPyConnection, lei: str, depth: int = 1,
                 max_nodes: int = MAX_NODES) -> dict:
    """The company plus everything within `depth` ownership/supplier links (either direction),
    breadth-first, capped at `max_nodes`."""
    tools = GraphTools(con)
    lei = tools._lei(lei)  # validates; raises ToolError
    edges = _edges(con)
    adj: dict[str, set[str]] = {}
    for e in edges:
        adj.setdefault(e["source"], set()).add(e["target"])
        adj.setdefault(e["target"], set()).add(e["source"])
    seen, queue, truncated = {lei: 0}, deque([lei]), False
    while queue:
        node = queue.popleft()
        if seen[node] >= depth:
            continue
        for nxt in sorted(adj.get(node, ())):
            if nxt in seen:
                continue
            if len(seen) >= max_nodes:
                truncated = True
                break
            seen[nxt] = seen[node] + 1
            queue.append(nxt)
    nodes = _nodes(con, list(seen))
    for n in nodes:
        n["hops"] = seen[n["id"]]
    return {"center": lei, "depth": depth, "truncated": truncated, "nodes": nodes,
            "edges": [e for e in edges if e["source"] in seen and e["target"] in seen]}


def scenario_graph(con: duckdb.DuckDBPyConnection, lei: str, severity: float = 1.0,
                   now: datetime | None = None, limit: int = 50) -> dict:
    """The disruption scenario as a subgraph: the shocked company, every impacted company, and
    the links along each impact path (oriented the way the risk travelled)."""
    result = GraphTools(con, now=now).run_scenario(lei, severity, limit)
    impacted, center = result["impacted"], result["ref"][4:]
    leis = {center} | {x for r in impacted for x in r["path_leis"]}
    nodes = _nodes(con, sorted(leis))
    delta = {r["ref"][4:]: r["increase"] for r in impacted}
    for n in nodes:
        n["delta"] = delta.get(n["id"], 0.0)
        n["shocked"] = n["id"] == center
    edges = {}
    for r in impacted:
        p = r["path_leis"]
        for (src, dst), rel in zip(pairwise(p), r["relations"], strict=True):
            edges[f"risk:{src}->{dst}"] = {"id": f"risk:{src}->{dst}", "source": src,
                                           "target": dst, "type": rel.split(":")[0],
                                           "relation": rel}
    return {"center": center, "severity": severity,
            "impacted": impacted, "nodes": nodes, "edges": list(edges.values())}
