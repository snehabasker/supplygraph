"""Multi-hop graph tools: parameterised, read-only Cypher over the Neo4j mirror.

The warehouse tools (`tools.py`) answer one-hop questions. Questions like "which of X's
suppliers, up to three tiers deep, sit within two hops of a sanctioned entity?" are path
queries, which is what the graph is for (docs/adr/0016). Each tool here is one fixed Cypher
query with variable-length paths. The model never writes Cypher: it picks a tool and passes
an LEI and a bounded depth.

Paths follow the same propagation rules as the risk engine (`risk/engine.py:SIGNAL_RELATIONS`):

* sanctions exposure travels ownership in either direction and supply links supplier -> customer;
* disruption travels supply links supplier -> customer and OWNED_BY child -> parent only.

Safety: every query runs in a READ transaction (Neo4j rejects writes in one), with a server-side
timeout and a row limit. Depths are validated integers, because Cypher cannot take a path length
as a parameter. With Neo4j Enterprise, point NEO4J_READ_USER at a reader-role account as well.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from supplygraph.agent.tools import LEI_PROP, ToolError, _schema

MAX_DEPTH = 4
DEFAULT_TIMEOUT = 5.0  # seconds, enforced by the server
ROW_CAP = 200

SUPPLY = "SUPPLIES|POSSIBLE_SUPPLIER"
CERTAINTY = {"SUPPLIES": "asserted", "POSSIBLE_SUPPLIER": "lead",
             "OWNED_BY": "asserted", "SANCTIONED_AS": "finding",
             "POSSIBLE_SANCTIONS_MATCH": "candidate"}

# A hop as returned to the model: direction is explicit, so the path reads unambiguously.
# (Braces doubled: the queries below go through str.format for their depth.)
_HOPS = """[r IN relationships(p) | {{from: startNode(r).lei, type: type(r), to: endNode(r).lei,
         relationship_type: r.relationship_type, accession: r.accession}}]"""

SUPPLIER_TREE = """
MATCH (root:Company {{lei: $lei}})
MATCH p = (s:Company)-[:{types}*1..{depth}]->(root)
WHERE s <> root
WITH s, p ORDER BY length(p)
WITH s, head(collect(p)) AS p
RETURN s.lei AS lei, s.legal_name AS name, s.legal_country AS country,
       s.risk_score AS risk_score, length(p) AS tier, """ + _HOPS + """ AS hops
ORDER BY tier, risk_score DESC, lei
LIMIT $cap
"""

# Quantified path pattern (Neo4j 5.9+): ownership in either direction, supply only forward
# (from the sanctioned side towards the company), checked hop by hop during expansion.
SANCTIONS_EXPOSURE = """
MATCH (root:Company {{lei: $lei}})
MATCH p = (src:Company)
          ((a)-[r:OWNED_BY|SUPPLIES|POSSIBLE_SUPPLIER]-(b)
           WHERE type(r) = 'OWNED_BY' OR startNode(r) = a){{0,{depth}}}
          (root)
MATCH (src)-[m:SANCTIONED_AS|POSSIBLE_SANCTIONS_MATCH]->(se:SanctionedEntity)
WITH se, src, m, p ORDER BY length(p), type(m) DESC
WITH se, head(collect({{src: src, m: m, p: p}})) AS best
WITH se, best.src AS src, best.m AS m, best.p AS p
RETURN se.id AS sanction_id, se.name AS sanction_name, type(m) AS match_type,
       m.match_method AS match_method, coalesce(m.confidence, m.score) AS confidence,
       src.lei AS via_lei, src.legal_name AS via_name, length(p) AS hops_to_company,
       """ + _HOPS + """ AS hops
ORDER BY hops_to_company, match_type DESC, sanction_id
LIMIT $cap
"""

DISRUPTION_REACH = """
MATCH (root:Company {{lei: $lei}})
MATCH p = (root)-[:{types}*1..{depth}]->(x:Company)
WHERE x <> root
WITH x, p ORDER BY length(p)
WITH x, head(collect(p)) AS p
RETURN x.lei AS lei, x.legal_name AS name, x.legal_country AS country,
       x.risk_score AS risk_score, length(p) AS hops, """ + _HOPS + """ AS path
ORDER BY hops, risk_score DESC, lei
LIMIT $cap
"""

ROOT = "MATCH (c:Company {lei: $lei}) RETURN c.legal_name AS name"

QUERIES = {"supplier_tree": SUPPLIER_TREE, "sanctions_exposure": SANCTIONS_EXPOSURE,
           "disruption_reach": DISRUPTION_REACH, "root": ROOT}

# Any clause that writes or reaches outside the graph. Tool queries are fixed strings; this is
# a guard against a future edit, checked in the unit tests.
WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL)\b", re.IGNORECASE)

DEPTH_PROP = {"type": "integer", "minimum": 1, "maximum": MAX_DEPTH}

CYPHER_TOOL_SPECS: list[dict] = [
    {"name": "supplier_tree",
     "description": "Multi-tier supply chain: every company that supplies this one directly "
                    "(tier 1) or through intermediate suppliers (tier 2+, up to depth), with "
                    "the link path and its filing refs. Each hop is 'asserted' (SUPPLIES) or a "
                    "'lead' (POSSIBLE_SUPPLIER). asserted_only=true drops leads.",
     "input_schema": _schema({"lei": LEI_PROP, "depth": DEPTH_PROP,
                              "asserted_only": {"type": "boolean"},
                              "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                             ["lei"])},
    {"name": "sanctions_exposure",
     "description": "Sanctioned entities within max_hops of this company along the paths "
                    "sanctions risk travels: ownership in either direction, and supply from "
                    "supplier to customer. Returns each sanctions record once with its "
                    "shortest path; match_type SANCTIONED_AS is a finding, "
                    "POSSIBLE_SANCTIONS_MATCH an unreviewed candidate. hops 0 = the company "
                    "itself.",
     "input_schema": _schema({"lei": LEI_PROP, "max_hops": DEPTH_PROP}, ["lei"])},
    {"name": "disruption_reach",
     "description": "Who is affected if this company is disrupted: companies reachable along "
                    "the paths disruption travels (supplier -> customer, subsidiary -> parent), "
                    "up to depth hops, each with its shortest path. Structural reach only; "
                    "run_scenario gives the risk-score impact.",
     "input_schema": _schema({"lei": LEI_PROP, "depth": DEPTH_PROP,
                              "asserted_only": {"type": "boolean"},
                              "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                             ["lei"])},
]
CYPHER_TOOLS = frozenset(s["name"] for s in CYPHER_TOOL_SPECS)


def _depth(value: Any, default: int) -> int:
    depth = default if value is None else value
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= MAX_DEPTH:
        raise ToolError(f"depth must be an integer from 1 to {MAX_DEPTH}")
    return depth


def _limit(value: Any, default: int = 25) -> int:
    limit = default if value is None else value
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ToolError("limit must be an integer from 1 to 100")
    return limit


def _hop(h: dict) -> dict:
    out = {"from_ref": f"lei:{h['from']}", "link": h["type"], "to_ref": f"lei:{h['to']}",
           "certainty": CERTAINTY.get(h["type"], "unknown")}
    if h.get("relationship_type"):
        out["link"] = f"OWNED_BY:{h['relationship_type']}"
    if h.get("accession"):
        out["filing_ref"] = f"filing:{h['accession']}"
    return out


def _weakest(hops: list[dict]) -> str:
    return "lead" if any(h["certainty"] == "lead" for h in hops) else "asserted"


class CypherTools:
    """Read-only path tools over a Neo4j driver (see module docstring)."""

    def __init__(self, driver: Any, *, database: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.driver = driver
        self.database = database
        self.timeout = timeout
        self.handlers: dict[str, Callable[..., dict]] = {
            "supplier_tree": self.supplier_tree,
            "sanctions_exposure": self.sanctions_exposure,
            "disruption_reach": self.disruption_reach,
        }

    # --- execution ----------------------------------------------------------------------

    def read(self, query: str, **params: Any) -> list[dict]:
        """Run one query in a READ transaction with a server-side timeout."""
        import neo4j
        from neo4j.exceptions import ClientError, Neo4jError, ServiceUnavailable

        if WRITE_CLAUSES.search(query):
            raise ToolError("refusing a query that is not read-only")

        @neo4j.unit_of_work(timeout=self.timeout)
        def work(tx: Any) -> list[dict]:
            return [record.data() for record in tx.run(query, **params)]

        try:
            with self.driver.session(database=self.database,
                                     default_access_mode=neo4j.READ_ACCESS) as session:
                return session.execute_read(work)
        except ServiceUnavailable as exc:
            raise ToolError("the graph database is unavailable; use the warehouse tools") \
                from exc
        except ClientError as exc:
            code = exc.code or ""
            if "TimedOut" in code or "Timeout" in code or "Terminated" in code:
                raise ToolError(f"graph query exceeded {self.timeout:g}s; use a smaller "
                                "depth") from exc
            raise
        except Neo4jError as exc:  # transient / database errors: report, don't crash the loop
            raise ToolError(f"graph query failed: {exc.code}") from exc

    def _root(self, lei: str) -> tuple[str, str | None]:
        lei = (lei or "").strip().upper()
        rows = self.read(ROOT, lei=lei)
        if not rows:
            raise ToolError(f"no company with LEI {lei!r} in the graph (use search_companies)")
        return lei, rows[0]["name"]

    # --- tools --------------------------------------------------------------------------

    def supplier_tree(self, lei: str, depth: int = 2, asserted_only: bool = False,
                      limit: int = 25) -> dict:
        depth, limit = _depth(depth, 2), _limit(limit)
        lei, name = self._root(lei)
        types = "SUPPLIES" if asserted_only else SUPPLY
        rows = self.read(SUPPLIER_TREE.format(types=types, depth=depth), lei=lei, cap=ROW_CAP)
        items = []
        for r in rows:
            hops = [_hop(h) for h in r["hops"]]
            items.append({"ref": f"lei:{r['lei']}", "name": r["name"], "country": r["country"],
                          "risk_score": r["risk_score"], "tier": r["tier"],
                          "certainty": _weakest(hops), "path": hops})
        return {"ref": f"lei:{lei}", "name": name, "depth": depth, "total": len(items),
                "suppliers": items[:limit],
                "note": "tier = supply hops to this company. certainty is the weakest link on "
                        "the path: 'lead' if any hop is a POSSIBLE_SUPPLIER lead."}

    def sanctions_exposure(self, lei: str, max_hops: int = 2) -> dict:
        depth = _depth(max_hops, 2)
        lei, name = self._root(lei)
        rows = self.read(SANCTIONS_EXPOSURE.format(depth=depth), lei=lei, cap=ROW_CAP)
        exposures = []
        for r in rows:
            hops = [_hop(h) for h in r["hops"]]
            exposures.append({
                "sanction_ref": f"sanction:{r['sanction_id']}",
                "sanction_name": r["sanction_name"], "match_type": r["match_type"],
                "certainty": CERTAINTY[r["match_type"]], "match_method": r["match_method"],
                "confidence": r["confidence"],
                "via": {"ref": f"lei:{r['via_lei']}", "name": r["via_name"]},
                "hops_to_company": r["hops_to_company"], "path": hops})
        return {"ref": f"lei:{lei}", "name": name, "max_hops": max_hops,
                "exposed": bool(exposures), "exposures": exposures,
                "note": "Exposure is structural: a sanctioned entity within reach along "
                        "ownership or supply links. It is not itself a sanctions finding about "
                        "this company."}

    def disruption_reach(self, lei: str, depth: int = 3, asserted_only: bool = False,
                         limit: int = 25) -> dict:
        depth, limit = _depth(depth, 3), _limit(limit)
        lei, name = self._root(lei)
        types = "SUPPLIES|OWNED_BY" if asserted_only else f"{SUPPLY}|OWNED_BY"
        rows = self.read(DISRUPTION_REACH.format(types=types, depth=depth), lei=lei,
                         cap=ROW_CAP)
        items = []
        for r in rows:
            hops = [_hop(h) for h in r["path"]]
            items.append({"ref": f"lei:{r['lei']}", "name": r["name"], "country": r["country"],
                          "risk_score": r["risk_score"], "hops": r["hops"],
                          "certainty": _weakest(hops), "path": hops})
        return {"ref": f"lei:{lei}", "name": name, "depth": depth, "total": len(items),
                "affected": items[:limit],
                "note": "Disruption follows supplier -> customer and subsidiary -> parent "
                        "links. Reach is not impact; see run_scenario for risk deltas."}
