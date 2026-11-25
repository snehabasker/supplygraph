# 0016. Multi-hop agent tools query Neo4j with Cypher; the tools are also served over MCP

**Status:** Accepted (M12). Supersedes the part of [0002](0002-duckdb-warehouse-neo4j-mirror.md)
that says every downstream consumer reads the warehouse. Extends
[0008](0008-agent-tools-grounded-citations.md) with three tools.

## Context
The agent's nine tools read DuckDB, and the only multi-hop one (`find_connection`) is a Python
breadth-first search. Questions such as "which of NVIDIA's suppliers, two tiers deep, are
leads?" and "is this company within two hops of a sanctioned entity?" are path queries. As
one-hop tools, the model has to chain calls and apply the propagation rules itself. Those
rules are direction-sensitive (sanctions travel ownership either way and supply forward;
disruption never travels parent → subsidiary), and the model can get them wrong. The
benchmark ([graph-bench.md](../graph-bench.md)) shows that the obvious alternatives,
similarity retrieval and a fixed-radius neighbourhood, either miss multi-hop answers or
flag companies the rules say are not exposed.

Other agents (Claude Code, Claude Desktop, agent frameworks) should be able to use the graph
without copying its queries.

## Decision
- **Three path tools in Cypher** (`agent/cypher.py`): `supplier_tree`, `sanctions_exposure`
  and `disruption_reach`. Each is one fixed, parameterised query with variable-length paths.
  `sanctions_exposure` uses a quantified path pattern that checks the direction of each hop
  during expansion. The model passes an LEI and a depth and never writes Cypher.
- **Read-only, bounded.** Each query runs in a READ transaction, which Neo4j refuses to write
  in (tested against the server). Each has a server-side timeout (`NEO4J_QUERY_TIMEOUT`,
  default 5 s), a row cap, and a depth that must be an integer from 1 to 4 (it is formatted
  into the pattern, because Cypher cannot take a path length as a parameter). A regex guard
  rejects write clauses before a query is sent. `NEO4J_READ_USER` selects a reader-role
  account where the edition has RBAC; Community has a single user, so there the READ
  session is the boundary.
- **Opt-in.** `ask --graph` and `mcp --graph` add the path tools when Neo4j is up. Without
  them the agent runs on the warehouse alone, as before, so unit tests and the offline eval
  still need no server. Neo4j errors become tool errors returned to the model, never a
  crash.
- **Same tool set everywhere.** `GraphTools.specs` lists exactly the tools an instance can
  run. The agent, the UI API and the MCP server all use it.
- **MCP server** (`supplygraph mcp`, stdio): the read-only tools with `readOnlyHint`. No
  proposal tools are exposed: an external agent cannot even propose an action into the
  audit log.
- Results keep their refs, so grounding checks (0008) cover the new tools unchanged.

## Alternatives considered
- **Let the model write Cypher (text-to-Cypher).** This is more flexible, but every query
  becomes untrusted input. Read-only sessions limit the damage but not cost or correctness,
  and the propagation rules would have to live in the prompt. It could be revisited as an
  explicit, separately evaluated tool.
- **Keep everything in Python over DuckDB.** This works at today's size (0002 already
  expected to move path queries to Neo4j as the graph grows) and is what `find_connection`
  still does. Writing the rules as Cypher path patterns keeps them declarative and is what
  the graph database is for.
- **Expose actions over MCP.** Rejected for now: proposals carry an evidence check that relies
  on the agent loop's record of retrieved refs, which an external client does not have.

## Consequences
- The agent's prompt changed (agent.system v2), and the tool descriptions are included in the
  prompt fingerprint.
- The path tools have two implementations of their semantics: the Cypher, and the
  benchmark's Python gold. The integration test and the benchmark check that they agree,
  which is what keeps them honest.
- The agent can now depend on Neo4j being loaded and fresh. `load-graph` runs after every
  build in `run-all` and in Dagster. A stale graph gives stale paths, not an error.
