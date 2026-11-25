"""Read-only retrieval tools over the warehouse (the graph's source of truth).

Every object a tool returns carries a `ref` (`lei:…`, `sanction:…`, `filing:…`, `news:…`).
The agent must cite refs, and `collect_refs` records which refs it actually retrieved, so an
answer citing something no tool returned is detectably ungrounded.

Text copied from filings, news and sanctions lists is returned as data inside JSON; the
system prompt tells the model it is untrusted.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any

import duckdb
from rapidfuzz import fuzz

from supplygraph.agent.actions import ActionStore
from supplygraph.er.normalize import normalize
from supplygraph.ingest.gleif import is_valid_lei
from supplygraph.warehouse import table_exists

MAX_TEXT = 300
RELATIONS = ("parents", "subsidiaries", "suppliers", "customers")


class ToolError(ValueError):
    """Bad tool input; the message goes back to the model as an error result."""


def _clip(text: str | None, limit: int = MAX_TEXT) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def collect_refs(value: Any) -> set[str]:
    """Every value stored under a key ending in `ref` / `refs` (structure, not text: a news
    title that *contains* "lei:…" does not count as retrieving it)."""
    out: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k.endswith("ref") and isinstance(v, str):
                out.add(v)
            elif k.endswith("refs") and isinstance(v, list):
                out.update(x for x in v if isinstance(x, str))
            else:
                out |= collect_refs(v)
    elif isinstance(value, list):
        for v in value:
            out |= collect_refs(v)
    return out


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


LEI_PROP = {"type": "string", "description": "20-character LEI, from search_companies."}

TOOL_SPECS: list[dict] = [
    {"name": "search_companies",
     "description": "Find companies by name, short name (e.g. 'TSMC') or LEI. Start here: every "
                    "other tool takes an LEI. Returns the best matches with their LEI and "
                    "risk score.",
     "input_schema": _schema({"query": {"type": "string"},
                              "limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                             ["query"])},
    {"name": "get_company",
     "description": "Profile of one company: names, country, status, watchlist membership, "
                    "risk score with its drivers (signal, source and link path), sanctions "
                    "matches, sanctions review-queue candidates, and analyst actions on it.",
     "input_schema": _schema({"lei": LEI_PROP}, ["lei"])},
    {"name": "get_relationships",
     "description": "A company's neighbours: 'parents' / 'subsidiaries' (GLEIF accounting "
                    "consolidation) or 'suppliers' / 'customers' (from SEC annual reports, with "
                    "the evidence sentence, filing accession and link type).",
     "input_schema": _schema({"lei": LEI_PROP,
                              "relation": {"type": "string", "enum": list(RELATIONS)},
                              "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                             ["lei", "relation"])},
    {"name": "get_news",
     "description": "Disruption-tagged news articles (GDELT) that name the company, newest "
                    "first. A mention is a signal, not proof the company was affected.",
     "input_schema": _schema({"lei": LEI_PROP,
                              "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                             ["lei"])},
    {"name": "get_sanction",
     "description": "One sanctions-list record (OpenSanctions): name, aliases, countries, "
                    "programs, source lists.",
     "input_schema": _schema({"sanction_id": {"type": "string"}}, ["sanction_id"])},
    {"name": "top_risks",
     "description": "Companies ranked by risk score. scope='own' ranks by risk from the "
                    "company's own signals, 'inherited' by risk arriving over links.",
     "input_schema": _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 50},
                              "scope": {"type": "string",
                                        "enum": ["total", "own", "inherited"]}}, [])},
    {"name": "review_queue",
     "description": "Unresolved company <-> sanctions-record candidates awaiting analyst "
                    "review (possible matches, not findings).",
     "input_schema": _schema({}, [])},
    {"name": "find_connection",
     "description": "Shortest chain of ownership / supplier links between two companies "
                    "(either direction, up to max_hops).",
     "input_schema": _schema({"from_lei": LEI_PROP, "to_lei": LEI_PROP,
                              "max_hops": {"type": "integer", "minimum": 1, "maximum": 6}},
                             ["from_lei", "to_lei"])},
    {"name": "run_scenario",
     "description": "What-if: the company is disrupted with the given severity (0-1). Returns "
                    "the companies whose risk rises most and the link path that carries it. "
                    "Disruption follows supplier links and subsidiary -> parent links only.",
     "input_schema": _schema({"lei": LEI_PROP,
                              "severity": {"type": "number", "minimum": 0, "maximum": 1},
                              "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                             ["lei"])},
]


class GraphTools:
    """The read-only tools: warehouse tools always; multi-hop Cypher tools (`cypher.py`) when a
    Neo4j-backed `CypherTools` is passed. Agent, UI and MCP server all go through this class."""

    def __init__(self, con: duckdb.DuckDBPyConnection, now: datetime | None = None,
                 store: ActionStore | None = None, cypher: Any | None = None) -> None:
        self.con = con
        self.now = now
        self.store = store  # live action state; else the `analyst_actions` table
        self.cypher = cypher
        self._names: dict[str, str] | None = None
        self.handlers: dict[str, Callable[..., dict]] = {
            "search_companies": self.search_companies, "get_company": self.get_company,
            "get_relationships": self.get_relationships, "get_news": self.get_news,
            "get_sanction": self.get_sanction, "top_risks": self.top_risks,
            "review_queue": self.review_queue, "find_connection": self.find_connection,
            "run_scenario": self.run_scenario,
        }
        if cypher is not None:
            self.handlers |= cypher.handlers

    @property
    def specs(self) -> list[dict]:
        """Tool specs for exactly the tools this instance can run."""
        if self.cypher is None:
            return list(TOOL_SPECS)
        from supplygraph.agent.cypher import CYPHER_TOOL_SPECS

        return TOOL_SPECS + CYPHER_TOOL_SPECS

    # --- dispatch ---------------------------------------------------------------------

    def call(self, name: str, arguments: dict) -> dict:
        handler = self.handlers.get(name)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        try:
            return handler(**arguments)
        except TypeError as exc:  # wrong / missing arguments
            raise ToolError(f"bad arguments for {name}: {exc}") from exc

    # --- helpers ----------------------------------------------------------------------

    def _rows(self, sql: str, params: list | None = None) -> list[dict]:
        cur = self.con.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    @property
    def names(self) -> dict[str, str]:
        if self._names is None:
            self._names = dict(self.con.execute("SELECT lei, legal_name FROM companies")
                               .fetchall())
        return self._names

    def _lei(self, lei: str) -> str:
        lei = (lei or "").strip().upper()
        if lei not in self.names:
            hint = "" if is_valid_lei(lei) else " (not a valid LEI; use search_companies)"
            raise ToolError(f"no company with LEI {lei!r}{hint}")
        return lei

    def _risk(self, lei: str) -> float | None:
        if not table_exists(self.con, "company_risk"):
            return None
        row = self.con.execute("SELECT risk_score FROM company_risk WHERE lei = ?",
                               [lei]).fetchone()
        return row[0] if row else None

    def _company_stub(self, lei: str) -> dict:
        return {"ref": f"lei:{lei}", "name": self.names.get(lei)}

    # --- tools ------------------------------------------------------------------------

    def search_companies(self, query: str, limit: int = 5) -> dict:
        q = (query or "").strip()
        if not q:
            raise ToolError("query is empty")
        if q.upper() in self.names:
            hits = {q.upper(): 100.0}
        else:
            target = normalize(q)
            hits: dict[str, float] = {}

            def consider(lei: str, name: str | None) -> None:
                if not name:
                    return
                n = normalize(name)
                if not n or not target:
                    score = 100.0 if name.casefold() == q.casefold() else 0.0
                else:
                    # Exact (normalized) names outrank fuzzy subset matches.
                    score = 100.0 if n == target else min(99.0, fuzz.token_set_ratio(target, n))
                if score > hits.get(lei, 0.0):
                    hits[lei] = score

            for lei, legal, other in self.con.execute(
                "SELECT lei, legal_name, other_names FROM companies"
            ).fetchall():
                for name in [legal, *(other or "").split(";")]:
                    consider(lei, name)
            if table_exists(self.con, "supplier_mentions"):  # short names from filings: "TSMC"
                for lei, written in self.con.execute(
                    "SELECT DISTINCT supplier_lei, name_as_written FROM supplier_mentions "
                    "WHERE supplier_lei IS NOT NULL"
                ).fetchall():
                    if lei in self.names:
                        consider(lei, written)
        ranked = sorted(((s, lei) for lei, s in hits.items() if s >= 70), reverse=True)
        rows = {r["lei"]: r for r in self._rows(
            "SELECT lei, legal_name, other_names, legal_country, entity_status FROM companies "
            "WHERE lei IN (SELECT unnest(?))", [[lei for _, lei in ranked[:limit]]])}
        return {"query": q, "matches": [
            {"ref": f"lei:{lei}", "name": rows[lei]["legal_name"],
             "other_names": _clip(rows[lei]["other_names"], 150),
             "country": rows[lei]["legal_country"], "status": rows[lei]["entity_status"],
             "risk_score": self._risk(lei), "match_score": round(score)}
            for score, lei in ranked[:limit]]}

    def get_company(self, lei: str) -> dict:
        lei = self._lei(lei)
        [c] = self._rows("SELECT * EXCLUDE (source) FROM companies WHERE lei = ?", [lei])
        out: dict[str, Any] = {
            "ref": f"lei:{lei}", "name": c["legal_name"], "other_names": c["other_names"],
            "registered_as": c["registered_as"], "legal_country": c["legal_country"],
            "hq_country": c["hq_country"], "legal_city": c["legal_city"],
            "entity_status": c["entity_status"],
        }
        if table_exists(self.con, "watchlist"):
            row = self.con.execute("SELECT reasons FROM watchlist WHERE lei = ?", [lei]).fetchone()
            out["watchlist"] = row[0].split(";") if row else []
        if table_exists(self.con, "company_risk"):
            rows = self._rows("SELECT * FROM company_risk WHERE lei = ?", [lei])
            if rows:
                r = rows[0]
                out["risk"] = {"score": r["risk_score"], "own": r["own_risk"],
                               "inherited": r["inherited_risk"],
                               "drivers": [self._driver(d) for d in json.loads(r["drivers"])]}
        out["sanctions_matches"] = [
            {"sanction_ref": f"sanction:{m['sanction_id']}", "sanction_name": m["name"],
             "match_method": m["match_method"], "confidence": m["confidence"],
             "evidence": _clip(m["evidence"])}
            for m in self._rows(
                "SELECT m.*, s.name FROM company_sanction_matches m "
                "LEFT JOIN sanctions s ON s.id = m.sanction_id WHERE m.lei = ?", [lei])]
        if table_exists(self.con, "er_candidates"):
            out["sanctions_candidates"] = [
                {"sanction_ref": f"sanction:{r['sanction_id']}", "sanction_name":
                 r["sanction_name"], "score": round(r["score"], 3),
                 "country_relation": r["country_relation"], "decision": r["decision"],
                 "method": r["method"], "reasoning": r["reasoning"]}
                for r in self._rows("SELECT * FROM er_candidates WHERE lei = ?", [lei])]
        if self.store is not None:
            out["analyst_actions"] = [
                {k: a.as_dict()[k] for k in ("action_id", "action", "sanction_id", "status",
                                             "reason", "proposed_by", "decided_by")}
                for a in self.store.actions() if a.lei == lei]
        elif table_exists(self.con, "analyst_actions"):
            out["analyst_actions"] = self._rows(
                "SELECT action_id, action, sanction_id, status, reason, proposed_by, "
                "decided_by FROM analyst_actions WHERE lei = ?", [lei])
        return out

    def _driver(self, d: dict) -> dict:
        """A stored risk driver, with refs for its source company and sanctions record."""
        out = {k: d.get(k) for k in ("signal", "detail", "value", "hops", "path", "relations")}
        out["detail"] = _clip(out["detail"], 200)
        out["source_ref"] = f"lei:{d['source']}"
        detail = d.get("detail") or ""
        for marker in (" match to ", "candidate "):
            if marker in detail:
                sid = detail.split(marker, 1)[1].split()[0]
                out["sanction_ref"] = f"sanction:{sid}"
        return out

    def get_relationships(self, lei: str, relation: str, limit: int = 25) -> dict:
        lei = self._lei(lei)
        if relation not in RELATIONS:
            raise ToolError(f"relation must be one of {RELATIONS}")
        if relation in ("parents", "subsidiaries"):
            mine, other = ("child_lei", "parent_lei") if relation == "parents" \
                else ("parent_lei", "child_lei")
            rows = self._rows(
                f"SELECT {other} AS lei, relationship_type FROM ownership WHERE {mine} = ? "
                f"ORDER BY relationship_type, {other}", [lei])
            items = [{**self._company_stub(r["lei"]),
                      "relationship_type": r["relationship_type"],
                      "country": None} for r in rows]
            for item in items:
                item["country"] = self.con.execute(
                    "SELECT legal_country FROM companies WHERE lei = ?",
                    [item["ref"][4:]]).fetchone()[0]
            note = "GLEIF accounting-consolidation parents (DIRECT = immediate, ULTIMATE = top)"
        else:
            if not table_exists(self.con, "supply_links"):
                return {"ref": f"lei:{lei}", "relation": relation, "total": 0, "items": [],
                        "note": "no supplier data loaded"}
            mine, other = ("customer_lei", "supplier_lei") if relation == "suppliers" \
                else ("supplier_lei", "customer_lei")
            rows = self._rows(
                f"SELECT {other} AS lei, link_type, extraction_method, supplies, evidence, "
                f"mentions, form, accession, filing_date FROM supply_links WHERE {mine} = ? "
                f"ORDER BY link_type DESC, mentions DESC, {other}", [lei])
            items = [{**self._company_stub(r["lei"]), "link_type": r["link_type"],
                      "extraction_method": r["extraction_method"], "supplies": r["supplies"],
                      "evidence": _clip(r["evidence"]), "mentions": r["mentions"],
                      "filing_ref": f"filing:{r['accession']}", "form": r["form"],
                      "filing_date": r["filing_date"]} for r in rows]
            note = ("SUPPLIES = asserted by the LLM extractor; POSSIBLE_SUPPLIER = a lead from "
                    "the rule-based extractor (~0.56 precision): it may be a competitor, "
                    "customer or partner")
        return {"ref": f"lei:{lei}", "name": self.names[lei], "relation": relation,
                "total": len(items), "items": items[:limit], "note": note}

    def get_news(self, lei: str, limit: int = 10) -> dict:
        lei = self._lei(lei)
        if not table_exists(self.con, "company_news"):
            return {"ref": f"lei:{lei}", "total": 0, "articles": []}
        rows = self._rows(
            "SELECT e.id, e.published_at, e.title, e.url, e.domain, e.event_types, "
            "e.copies, n.matched_name FROM company_news n JOIN news_events e "
            "ON e.id = n.event_id WHERE n.lei = ? ORDER BY e.published_at DESC, e.id", [lei])
        return {"ref": f"lei:{lei}", "name": self.names[lei], "total": len(rows),
                "articles": [{"ref": f"news:{r['id']}", "published_at": _iso(r["published_at"]),
                              "title": _clip(r["title"], 200), "url": r["url"],
                              "domain": r["domain"], "event_types": r["event_types"],
                              "copies": r["copies"], "matched_name": r["matched_name"]}
                             for r in rows[:limit]],
                "note": "Event types come from GDELT theme tags and are often loose; a "
                        "mention is not evidence of an actual disruption."}

    def get_sanction(self, sanction_id: str) -> dict:
        sid = (sanction_id or "").strip()
        if sid.startswith("sanction:"):
            sid = sid[len("sanction:"):]
        rows = self._rows("SELECT * FROM sanctions WHERE id = ?", [sid])
        if not rows:
            raise ToolError(f"no sanctions record {sid!r}")
        s = rows[0]
        aliases = [a for a in (s["aliases"] or "").split(";") if a]
        return {"ref": f"sanction:{sid}", "name": s["name"], "schema": s["schema"],
                "aliases": aliases[:20], "alias_count": len(aliases),
                "countries": s["countries"], "program_ids": s["program_ids"],
                "lists": _clip(s["dataset"], 400), "sanctions": _clip(s["sanctions"], 600),
                "last_seen": s["last_seen"]}

    def top_risks(self, limit: int = 10, scope: str = "total") -> dict:
        column = {"total": "risk_score", "own": "own_risk", "inherited": "inherited_risk"}.get(
            scope)
        if column is None:
            raise ToolError("scope must be total, own or inherited")
        rows = self._rows(
            f"SELECT r.lei, c.legal_name, c.legal_country, r.risk_score, r.own_risk, "
            f"r.inherited_risk, r.drivers FROM company_risk r JOIN companies c USING (lei) "
            f"WHERE r.{column} > 0 ORDER BY r.{column} DESC, r.lei LIMIT ?", [limit])
        return {"scope": scope, "companies": [
            {"ref": f"lei:{r['lei']}", "name": r["legal_name"], "country": r["legal_country"],
             "risk_score": r["risk_score"], "own_risk": r["own_risk"],
             "inherited_risk": r["inherited_risk"],
             "top_driver": (self._driver(json.loads(r["drivers"])[0])
                            if r["drivers"] and json.loads(r["drivers"]) else None)}
            for r in rows]}

    def review_queue(self) -> dict:
        if not table_exists(self.con, "er_candidates"):
            return {"candidates": []}
        rows = self._rows("SELECT * FROM er_candidates WHERE decision = 'review' "
                          "ORDER BY score DESC, lei")
        return {"candidates": [
            {"company_ref": f"lei:{r['lei']}", "company_name": r["company_name"],
             "sanction_ref": f"sanction:{r['sanction_id']}", "sanction_name": r["sanction_name"],
             "score": round(r["score"], 3), "name_score": round(r["name_score"], 3),
             "country_relation": r["country_relation"]} for r in rows],
            "note": "Candidates are possible matches pending analyst review, not findings."}

    def _adjacency(self) -> dict[str, list[tuple[str, str, str | None]]]:
        adj: dict[str, list[tuple[str, str, str | None]]] = {}
        for child, parent, rtype in self.con.execute(
            "SELECT child_lei, parent_lei, relationship_type FROM ownership"
        ).fetchall():
            adj.setdefault(child, []).append((parent, f"OWNED_BY:{rtype} (child -> parent)",
                                              None))
            adj.setdefault(parent, []).append((child, f"PARENT_OF:{rtype} (parent -> child)",
                                               None))
        if table_exists(self.con, "supply_links"):
            for sup, cust, ltype, acc in self.con.execute(
                "SELECT supplier_lei, customer_lei, link_type, accession FROM supply_links"
            ).fetchall():
                adj.setdefault(sup, []).append((cust, f"{ltype} (supplier -> customer)", acc))
                adj.setdefault(cust, []).append((sup, f"{ltype} (customer <- supplier)", acc))
        return adj

    def find_connection(self, from_lei: str, to_lei: str, max_hops: int = 4) -> dict:
        start, goal = self._lei(from_lei), self._lei(to_lei)
        adj = self._adjacency()
        prev: dict[str, tuple[str, str, str | None] | None] = {start: None}
        queue, depth = deque([start]), {start: 0}
        while queue and goal not in prev:
            node = queue.popleft()
            if depth[node] >= max_hops:
                continue
            for nxt, rel, acc in sorted(adj.get(node, [])):
                if nxt not in prev:
                    prev[nxt], depth[nxt] = (node, rel, acc), depth[node] + 1
                    queue.append(nxt)
        if goal not in prev:
            return {"from_ref": f"lei:{start}", "to_ref": f"lei:{goal}", "connected": False,
                    "note": f"no ownership/supplier chain within {max_hops} hops"}
        hops, node = [], goal
        while prev[node] is not None:
            parent, rel, acc = prev[node]
            hop = {"from": self._company_stub(parent), "to": self._company_stub(node),
                   "link": rel}
            if acc:
                hop["filing_ref"] = f"filing:{acc}"
            hops.append(hop)
            node = parent
        return {"from_ref": f"lei:{start}", "to_ref": f"lei:{goal}", "connected": True,
                "hops": list(reversed(hops))}

    def run_scenario(self, lei: str, severity: float = 1.0, limit: int = 10) -> dict:
        from supplygraph.risk import engine

        lei = self._lei(lei)
        if not 0 <= severity <= 1:
            raise ToolError("severity must be between 0 and 1")
        impacted = engine.scenario(self.con, lei, severity, now=self.now)
        return {"ref": f"lei:{lei}", "name": self.names[lei], "severity": severity,
                "companies_affected": len(impacted), "impacted": [
                    {"ref": f"lei:{r['lei']}", "name": r["name"], "risk_before": r["before"],
                     "risk_after": r["after"], "increase": r["delta"], "path": r["path"],
                     "path_leis": r["path_leis"], "relations": r["relations"]}
                    for r in impacted[:limit]]}
