"""Multi-hop retrieval benchmark: vector RAG vs GraphRAG vs agent tools with Cypher.

Question set (`build_questions`): three families of multi-hop questions, generated from the
warehouse, each with a **gold answer set computed in SQL/Python from the warehouse tables**,
independently of the Cypher under test:

* `supplier_tiers`     companies that supply X directly or via one intermediate (depth 2);
* `disruption_reach`   companies a disruption at X reaches within 3 hops, along supplier ->
                       customer and subsidiary -> parent links (the risk engine's rules);
* `sanctions_exposure` sanctions records (findings and review candidates) within 2 hops of X,
                       along ownership (either direction) and supplier -> customer links.
                       Includes as many unexposed companies as exposed ones (gold = empty).

Methods (`evaluate`), all given the same question text:

* `vector_rag`  TF-IDF cosine over text chunks (one per company, ownership link, supply link
                with its evidence sentence, sanctions match), top-k. A lexical vector baseline:
                no neural embedder, so no model download and fully deterministic.
* `graph_rag`   entity-link the company named in the question (`search_companies`), then pull
                its radius-r neighbourhood from Neo4j and serialise it as triples (GraphRAG
                "local search").
* `cypher_tools` entity-link, then call the one path tool for the question family
                (`supplier_tree` / `disruption_reach` / `sanctions_exposure`). Tool choice is
                **oracle** (by family), so this is the upper bound of an agent that picks the
                right tool; `evaluate-agent` measures the model's own tool choice.

What is scored is the *retrieval stage*: the set of answer-type refs (companies, or sanctions
records for exposure) present in what each method hands the model, against gold — recall (did
the context contain the answer?), precision (how much must the model filter out?), exact set
match, context size and latency. No LLM is called; the end-to-end answer accuracy needs API
credit and is reported separately when run.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb

from supplygraph.warehouse import table_exists

FAMILIES = {
    "supplier_tiers": {"tool": "supplier_tree", "arg": "depth", "depth": 2, "answer": "lei",
                       "template": "Which companies supply {name}, either directly or through "
                                   "one intermediate supplier (up to two tiers)?"},
    "disruption_reach": {"tool": "disruption_reach", "arg": "depth", "depth": 3,
                         "answer": "lei",
                         "template": "If {name} were disrupted, which companies could be "
                                     "affected within three hops: its customers, their "
                                     "customers, and parent companies?"},
    "sanctions_exposure": {"tool": "sanctions_exposure", "arg": "max_hops", "depth": 2,
                           "answer": "sanction",
                           "template": "Is {name} within two ownership or supply hops of a "
                                       "sanctioned entity or a sanctions candidate? Name the "
                                       "sanctions records."},
}
METHODS = ("vector_rag", "graph_rag", "cypher_tools")


# --- the graph, from the warehouse (gold side) -----------------------------------------------

class WarehouseGraph:
    """Edges and sanctions matches read straight from the warehouse tables."""

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.supply = con.execute(
            "SELECT DISTINCT supplier_lei, customer_lei FROM supply_links"
        ).fetchall() if table_exists(con, "supply_links") else []
        self.ownership = con.execute(
            "SELECT DISTINCT child_lei, parent_lei FROM ownership").fetchall()
        self.matches: dict[str, set[str]] = defaultdict(set)
        for lei, sid in con.execute("SELECT lei, sanction_id FROM company_sanction_matches"
                                    ).fetchall():
            self.matches[lei].add(sid)
        if table_exists(con, "er_candidates"):  # the review queue = POSSIBLE_SANCTIONS_MATCH
            for lei, sid in con.execute("SELECT lei, sanction_id FROM er_candidates "
                                        "WHERE decision = 'review'").fetchall():
                self.matches[lei].add(sid)

    def _adj(self, pairs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = defaultdict(set)
        for a, b in pairs:
            if a != b:
                adj[a].add(b)
        return adj

    def suppliers_of(self) -> dict[str, set[str]]:  # customer -> suppliers
        return self._adj((c, s) for s, c in self.supply)

    def disruption_next(self) -> dict[str, set[str]]:  # supplier -> customer, child -> parent
        return self._adj([*self.supply, *self.ownership])

    def exposure_prev(self) -> dict[str, set[str]]:
        """y -> x where risk can move x -> y: ownership either way, supply forward. Walking it
        from a company finds the companies whose sanctions reach it."""
        return self._adj([*((c, s) for s, c in self.supply), *self.ownership,
                          *((p, c) for c, p in self.ownership)])


def reach(start: str, nxt: dict[str, set[str]], depth: int) -> dict[str, int]:
    """Nodes reachable from `start` in 1..depth hops -> their shortest hop count."""
    seen, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        if seen[node] >= depth:
            continue
        for n in sorted(nxt.get(node, ())):
            if n not in seen:
                seen[n] = seen[node] + 1
                queue.append(n)
    seen.pop(start)
    return seen


def gold_answer(g: WarehouseGraph, family: str, lei: str) -> tuple[list[str], int]:
    """(sorted gold refs, hops needed = the longest shortest path among them)."""
    depth = FAMILIES[family]["depth"]
    if family == "supplier_tiers":
        hits = reach(lei, g.suppliers_of(), depth)
        return sorted(f"lei:{x}" for x in hits), max(hits.values(), default=0)
    if family == "disruption_reach":
        hits = reach(lei, g.disruption_next(), depth)
        return sorted(f"lei:{x}" for x in hits), max(hits.values(), default=0)
    hits = {lei: 0, **reach(lei, g.exposure_prev(), depth)}
    best: dict[str, int] = {}
    for company, hops in hits.items():
        for sid in g.matches.get(company, ()):
            best[sid] = min(hops, best.get(sid, hops))
    return sorted(f"sanction:{s}" for s in best), max(best.values(), default=0)


# --- question set ----------------------------------------------------------------------------

def display_names(con: duckdb.DuckDBPyConnection, seeds: Iterable[Path] = ()) -> dict[str, str]:
    """Names to put in questions: the seed-file name where there is one (e.g. "NVIDIA
    Corporation" rather than a Chinese legal name for TSMC), else the legal name."""
    names = dict(con.execute("SELECT lei, legal_name FROM companies").fetchall())
    for path in seeds:
        if path.exists():
            import csv

            with path.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("lei") in names and row.get("name"):
                        names[row["lei"]] = row["name"]
    return names


def build_questions(con: duckdb.DuckDBPyConnection, names: dict[str, str]) -> list[dict]:
    g = WarehouseGraph(con)
    subjects: dict[str, list[str]] = {
        "supplier_tiers": sorted({c for _, c in g.supply}),
        "disruption_reach": sorted({s for s, _ in g.supply}),
    }
    exposed, unexposed = [], []
    candidates = sorted({x for pair in [*g.supply, *g.ownership] for x in pair})
    for lei in candidates:
        (exposed if gold_answer(g, "sanctions_exposure", lei)[0] else unexposed).append(lei)
    # Hard negatives: unexposed companies nearest a sanctioned one in the *undirected* graph
    # (e.g. a supplier of a sanctioned company's customer: close, but sanctions risk does not
    # flow customer -> supplier), then by LEI; as many as there are positives.
    undirected = g._adj([*g.supply, *((c, s) for s, c in g.supply), *g.ownership,
                         *((p, c) for c, p in g.ownership)])

    def distance(lei: str) -> int:
        near = reach(lei, undirected, 4)
        return min((h for x, h in near.items() if g.matches.get(x)), default=99)

    unexposed.sort(key=lambda lei: (distance(lei), lei))
    subjects["sanctions_exposure"] = sorted(exposed + unexposed[:len(exposed)])

    questions = []
    for family, leis in subjects.items():
        spec = FAMILIES[family]
        for lei in leis:
            gold, hops = gold_answer(g, family, lei)
            questions.append({
                "id": f"{family}-{lei}", "family": family, "lei": lei, "name": names[lei],
                "question": spec["template"].format(name=names[lei]),
                "gold": gold, "hops_needed": hops})
    return questions


def write_questions(questions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")


def read_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stale_gold(questions: list[dict], con: duckdb.DuckDBPyConnection) -> list[str]:
    """Questions whose committed gold no longer matches the warehouse."""
    g = WarehouseGraph(con)
    return [q["id"] for q in questions
            if gold_answer(g, q["family"], q["lei"])[0] != q["gold"]]


# --- method 1: vector RAG --------------------------------------------------------------------

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    words = [w.casefold() for w in _WORD.findall(text)]
    return words + [f"{a} {b}" for a, b in pairwise(words)]


class VectorIndex:
    """TF-IDF (unigrams + bigrams, sublinear tf) with cosine similarity."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        docs = [Counter(_tokens(c["text"])) for c in chunks]
        df = Counter(t for d in docs for t in d)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + k)) + 1 for t, k in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(k)) * self.idf.get(t, 0.0) for t, k in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items() if x}

    def search(self, query: str, k: int) -> list[dict]:
        q = self._vec(Counter(_tokens(query)))
        scored = [(sum(w * v.get(t, 0.0) for t, w in q.items()), i)
                  for i, v in enumerate(self.vecs)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.chunks[i] for s, i in scored[:k] if s > 0]


def vector_chunks(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """The text a vector store would hold, each chunk tagged with the refs it is about."""
    names = dict(con.execute("SELECT lei, legal_name FROM companies").fetchall())
    chunks = []
    for lei, legal, other, country, status in con.execute(
        "SELECT lei, legal_name, other_names, legal_country, entity_status FROM companies "
        "ORDER BY lei"
    ).fetchall():
        aka = f" (also known as {other.replace(';', ', ')})" if other else ""
        chunks.append({"text": f"{legal}{aka} is a company registered in {country}, LEI {lei}, "
                               f"status {status}.", "refs": [f"lei:{lei}"]})
    for child, parent, rtype in con.execute(
        "SELECT child_lei, parent_lei, relationship_type FROM ownership ORDER BY ALL"
    ).fetchall():
        chunks.append({"text": f"{names.get(child)} is a subsidiary of {names.get(parent)} "
                               f"({rtype.lower()} parent).",
                       "refs": [f"lei:{child}", f"lei:{parent}"]})
    if table_exists(con, "supply_links"):
        for sup, cust, ltype, evidence in con.execute(
            "SELECT supplier_lei, customer_lei, link_type, evidence FROM supply_links "
            "ORDER BY ALL"
        ).fetchall():
            kind = "a supplier" if ltype == "SUPPLIES" else "a possible supplier"
            chunks.append({"text": f"The annual report of {names.get(cust)} names "
                                   f"{names.get(sup)} as {kind}: \"{evidence or ''}\"",
                           "refs": [f"lei:{sup}", f"lei:{cust}"]})
    rows = con.execute(
        "SELECT m.lei, m.sanction_id, s.name, 'a sanctions match (finding)' FROM "
        "company_sanction_matches m LEFT JOIN sanctions s ON s.id = m.sanction_id").fetchall()
    if table_exists(con, "er_candidates"):
        rows += con.execute(
            "SELECT lei, sanction_id, sanction_name, 'a possible sanctions match awaiting "
            "review' FROM er_candidates WHERE decision = 'review'").fetchall()
    for lei, sid, sname, what in sorted(rows):
        chunks.append({"text": f"{names.get(lei)} has {what} against the sanctions record "
                               f"{sname} ({sid}).",
                       "refs": [f"lei:{lei}", f"sanction:{sid}"]})
    return chunks


# --- methods 2 and 3: graph ------------------------------------------------------------------

NEIGHBOURHOOD = """
MATCH (seed:Company {{lei: $lei}})
MATCH p = (seed)-[*1..{radius}]-(n)
UNWIND relationships(p) AS r
WITH DISTINCT r
RETURN startNode(r).lei AS a_lei, startNode(r).id AS a_id,
       coalesce(startNode(r).legal_name, startNode(r).name, startNode(r).title) AS a_name,
       type(r) AS type,
       endNode(r).lei AS b_lei, endNode(r).id AS b_id,
       coalesce(endNode(r).legal_name, endNode(r).name, endNode(r).title) AS b_name
LIMIT $cap
"""


def _node_ref(lei: str | None, nid: str | None, rel_type: str) -> str | None:
    if lei:
        return f"lei:{lei}"
    if nid and rel_type in ("SANCTIONED_AS", "POSSIBLE_SANCTIONS_MATCH"):
        return f"sanction:{nid}"
    return f"news:{nid}" if nid else None


def neighbourhood(cypher: Any, lei: str, radius: int, cap: int = 2000) -> tuple[str, set[str]]:
    rows = cypher.read(NEIGHBOURHOOD.format(radius=int(radius)), lei=lei, cap=cap)
    lines, refs = [], set()
    for r in rows:
        a = _node_ref(r["a_lei"], r["a_id"], r["type"])
        b = _node_ref(r["b_lei"], r["b_id"], r["type"])
        refs |= {x for x in (a, b) if x}
        lines.append(f"({r['a_name']} [{a}]) -[{r['type']}]-> ({r['b_name']} [{b}])")
    return "\n".join(sorted(lines)), refs


# --- scoring ---------------------------------------------------------------------------------

def _answer_refs(refs: Iterable[str], family: str, subject: str) -> set[str]:
    kind = FAMILIES[family]["answer"]
    return {r for r in refs if r.startswith(f"{kind}:") and r != f"lei:{subject}"}


def score(gold: list[str], predicted: set[str]) -> dict:
    """recall is undefined with no gold (a negative question), precision with no prediction;
    a negative question that gets any prediction is a false alarm."""
    g = set(gold)
    hit = len(g & predicted)
    return {"recall": hit / len(g) if g else None,
            "precision": hit / len(predicted) if predicted else None,
            "exact": predicted == g, "false_alarm": (not g and bool(predicted)),
            "predicted": len(predicted)}


def _tokens_estimate(text: str) -> int:
    return math.ceil(len(text) / 4)


def evaluate(questions: list[dict], con: duckdb.DuckDBPyConnection, cypher: Any,
             *, k: int = 10, radius: int = 2,
             progress: Callable[[dict], None] | None = None) -> dict:
    """Run every method on every question; returns {"summary", "by_family", "rows"}."""
    from supplygraph.agent.tools import GraphTools, ToolError

    tools = GraphTools(con, cypher=cypher)
    index = VectorIndex(vector_chunks(con))
    rows = []
    for q in questions:
        fam = FAMILIES[q["family"]]
        row = {"id": q["id"], "family": q["family"], "hops_needed": q["hops_needed"],
               "gold": len(q["gold"])}

        t0 = time.perf_counter()
        hits = index.search(q["question"], k)
        ms = (time.perf_counter() - t0) * 1000
        context = "\n".join(h["text"] for h in hits)
        refs = {r for h in hits for r in h["refs"]}
        row["vector_rag"] = {**score(q["gold"], _answer_refs(refs, q["family"], q["lei"])),
                             "tokens": _tokens_estimate(context), "ms": ms}

        t0 = time.perf_counter()
        matches = tools.search_companies(q["name"], limit=1)["matches"]
        linked = matches[0]["ref"][4:] if matches else None
        link_ms = (time.perf_counter() - t0) * 1000
        row["linked"] = linked == q["lei"]

        if linked is None:
            for m in ("graph_rag", "cypher_tools"):
                row[m] = {**score(q["gold"], set()), "tokens": 0, "ms": link_ms}
        else:
            t0 = time.perf_counter()
            context, refs = neighbourhood(cypher, linked, radius)
            ms = link_ms + (time.perf_counter() - t0) * 1000
            row["graph_rag"] = {**score(q["gold"], _answer_refs(refs, q["family"], linked)),
                                "tokens": _tokens_estimate(context), "ms": ms}

            t0 = time.perf_counter()
            args = {"lei": linked, fam["arg"]: fam["depth"]}
            if q["family"] != "sanctions_exposure":
                args["limit"] = 100
            try:
                result = tools.call(fam["tool"], args)
                text = json.dumps(result, ensure_ascii=False, default=str)
                # The answer list itself, not every ref on the paths behind it.
                refs = ({e["sanction_ref"] for e in result["exposures"]}
                        if q["family"] == "sanctions_exposure" else
                        {i["ref"] for i in result.get("suppliers", result.get("affected", []))})
            except ToolError:
                text, refs = "", set()
            ms = link_ms + (time.perf_counter() - t0) * 1000
            row["cypher_tools"] = {
                **score(q["gold"], _answer_refs(refs, q["family"], linked)),
                "tokens": _tokens_estimate(text), "ms": ms}
        rows.append(row)
        if progress:
            progress(row)
    return {"params": {"k": k, "radius": radius, "questions": len(rows)},
            "summary": summarize(rows),
            "by_family": {f: summarize([r for r in rows if r["family"] == f])
                          for f in FAMILIES},
            "multi_hop": summarize([r for r in rows if r["hops_needed"] >= 2]),
            "rows": rows}


def summarize(rows: list[dict]) -> dict:
    """Means over the questions where each metric is defined (see `score`)."""
    negatives = sum(1 for r in rows if r["gold"] == 0)
    out: dict[str, Any] = {"questions": len(rows), "negatives": negatives}
    if not rows:
        return out

    def mean(values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return round(statistics.fmean(vals), 3) if vals else None

    for m in METHODS:
        vals = [r[m] for r in rows]
        out[m] = {
            "recall": mean([v["recall"] for v in vals]),
            "precision": mean([v["precision"] for v in vals]),
            "exact": round(sum(v["exact"] for v in vals) / len(vals), 3),
            "false_alarm": (round(sum(v["false_alarm"] for v in vals) / negatives, 3)
                            if negatives else None),
            "tokens_median": int(statistics.median(v["tokens"] for v in vals)),
            "ms_median": round(statistics.median(v["ms"] for v in vals), 1),
        }
    return out
