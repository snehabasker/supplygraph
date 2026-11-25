# Multi-hop retrieval benchmark: vector RAG vs GraphRAG vs Cypher tools (Milestone 12)

**Question:** for multi-hop supply-chain questions, what does each retrieval strategy put in
front of the model? Does the context contain the answer, and how much else must the model
filter out?

```powershell
supplygraph load-graph                         # Neo4j must mirror the warehouse
supplygraph build-graph-bench                  # questions + gold -> eval/graph_bench/questions.jsonl
supplygraph evaluate-graph-bench [--radius 3] [--track]
```

## Question set

There are 63 questions, generated from the live warehouse (278 companies, 239 ownership links,
30 supply links, 1 sanctions finding, 2 review candidates). Each question names one company and
has a gold answer set. The gold is **computed in Python from the warehouse tables**
(`graph_bench.gold_answer`), not by the Cypher under test.

| Family | n | Gold answer | Rule |
|---|---|---|---|
| `supplier_tiers` | 9 | companies supplying X directly or via one intermediate | supply links, backwards, depth 2 |
| `disruption_reach` | 22 | companies a disruption at X reaches within 3 hops | supplier → customer and subsidiary → parent only |
| `sanctions_exposure` | 32 | sanctions records (findings + review candidates) within 2 hops | ownership either way, supplier → customer |

The rules are the risk engine's `SIGNAL_RELATIONS`, so the benchmark asks the questions the
product's risk scores are built on.

- **Multi-hop subset:** 22 questions where at least one gold answer is 2+ hops away.
- **Negatives:** half of the exposure questions (16) have no answer. They are *hard
  negatives*: unexposed companies chosen by undirected distance to a sanctioned company. Carl
  Zeiss is an example. It supplies ASML, whose Hong Kong subsidiary is a sanctions candidate,
  so it is close in the graph. Sanctions risk does not flow from a customer back to its
  suppliers, though.
- **Stale gold:** `evaluate-graph-bench` refuses to run when the committed gold no longer
  matches the warehouse.

## Methods

All three methods get the same question text.

- **`vector_rag`** retrieves the top 10 chunks by TF-IDF cosine similarity (unigrams and
  bigrams). There is one chunk per company, ownership link, supply link (with its evidence
  sentence) and sanctions match. This is a lexical vector baseline: deterministic, with no
  model download.
- **`graph_rag`** entity-links the named company (`search_companies`), pulls its radius-r
  neighbourhood from Neo4j over all link types, and serialises it as triples. This is
  GraphRAG "local search".
- **`cypher_tools`** entity-links the company, then calls the path tool for the question
  family (`supplier_tree`, `disruption_reach` or `sanctions_exposure`). The tool is chosen by
  family (**oracle routing**), so this method is the upper bound of an agent that picks the
  right tool.

**Scoring.** A method's prediction is the set of answer-type refs in its context: companies,
or sanctions records for exposure questions. The subject company itself is excluded.

- **Recall:** whether the context contains the gold. It is averaged over questions that have
  an answer.
- **Precision:** how much of the context is the answer. It is averaged over questions where
  the method returned something.
- **Exact:** whether the set equals the gold.
- **False alarm:** whether a no-answer question got any prediction.

Tokens are estimated as characters ÷ 4. Latency is measured locally and includes entity
linking (~40 ms of the graph methods' time).

## Results (2026-11-25, live data)

| Method | Recall | Precision | Exact | False alarm (16 neg.) | Context tokens (median) |
|---|---|---|---|---|---|
| vector RAG (k=10) | 0.707 | 0.174 | 0.000 | 1.000 | 418 |
| GraphRAG, radius 2 | 0.755 | 0.319 | 0.317 | 0.000 | 408 |
| GraphRAG, radius 3 | 1.000 | 0.376 | 0.460 | 0.188 | 982 |
| **Cypher tools** | **1.000** | **1.000** | **1.000** | **0.000** | **186** |

Multi-hop subset (22 questions):

| Method | Recall | Precision | Tokens |
|---|---|---|---|
| vector RAG | 0.737 | 0.308 | 389 |
| GraphRAG r=2 | 0.477 | 0.443 | 408 |
| GraphRAG r=3 | 1.000 | 0.604 | 982 |
| Cypher tools | 1.000 | 1.000 | 247 |

By family (recall / precision):

| Family | vector RAG | GraphRAG r=2 | GraphRAG r=3 | Cypher tools |
|---|---|---|---|---|
| supplier tiers | 0.299 / 0.176 | 1.000 / 0.261 | 1.000 / 0.116 (1 883 tokens) | 1.000 / 1.000 |
| disruption reach | 0.706 / 0.146 | 1.000 / 0.188 | 1.000 / 0.081 | 1.000 / 1.000 |
| sanctions exposure | 0.938 / 0.194 | 0.281 / 1.000 | 1.000 / 0.842 | 1.000 / 1.000 |

## What the failures are

- **Vector RAG misses tier-2 suppliers.** It finds 30% of them. A tier-2 supplier's name
  appears in no chunk that resembles the question. The question names NVIDIA, so the
  retriever returns NVIDIA's own supply chunks and subsidiaries, but not its suppliers'
  suppliers. A neural embedder would not change this: the missing hop is structural, not
  lexical.
- **Vector RAG raises a false alarm on every no-answer sanctions question.** The words
  "sanctioned" and "sanctions record" match all three sanctions-match chunks every time. On
  this small corpus that also makes its recall look high (0.94). The context is right for
  the wrong reason, and an LLM would have to work out exposure from scattered facts.
- **GraphRAG's radius is a trade-off with no right setting.** At r=2 it cannot see a
  sanctioned company 2 hops away, because the `SanctionedEntity` node is a third hop
  (recall 0.28). At r=3 it finds everything, but the context is 5× the tool output and
  flags 3 of the 16 hard negatives. Its neighbourhood ignores link direction, so Carl
  Zeiss, Carl Zeiss SMT and TSMC look exposed through ASML even though sanctions risk does
  not travel that way.
- **Cypher tools are exact because the path rules are in the query.** Direction and link
  type are part of the pattern, including a quantified path pattern for "ownership either
  way, supply forward". Their output is also the smallest context.

## Caveats (read before quoting the numbers)

1. **The Cypher tools' perfect score is partly by construction.** The tools implement the
   same rules the gold encodes, and the routing is an oracle. What 63/63 does show is that
   the Cypher and the independent Python implementation agree on every question. The
   integration test checks the same on a synthetic graph. It does **not** show that an LLM
   picks the right tool.
2. **Retrieval only.** No LLM was called. End-to-end answer accuracy (`supplygraph ask
   --graph`, `evaluate-agent`) needs API credit and has not been measured.
3. **Small, shallow graph.** There are 30 supply links, all of them `POSSIBLE_SUPPLIER`
   leads, because LLM extraction is unrun. Ownership is at most 2 levels deep and there are
   3 sanctions records. With 74 000 sanctions records in the vector corpus, vector RAG's
   false-alarm pattern would change. Its multi-hop miss would not.
4. **The vector baseline is TF-IDF.** It is fair on names, which dominate these questions,
   but it is not a neural embedding model.

## Next

- End-to-end: run the agent with `--graph` on these questions and grade its cited refs
  against the gold (pending API credit).
- Add the multi-hop questions to the agent eval cases once a Neo4j-backed fixture run exists.
