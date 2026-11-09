# 0002. DuckDB is the source of truth; Neo4j mirrors it

**Status:** Accepted (M1; reaffirmed in M6 and M7)

## Context
The data arrives as files (GLEIF JSON, a sanctions CSV, SEC filings, GDELT dumps) and must be
cleaned, joined, resolved and checked before anyone sees it. The product questions, though, are
graph questions: who owns whom, and what does a disruption reach.

## Decision
A DuckDB file (`data/warehouse.duckdb`) is the canonical store. Every pipeline step reads and
writes it with SQL, and quality checks run against it. Neo4j is loaded from it (`load-graph`,
MERGE, with pruning scoped to the loaded companies) for traversal queries and the Neo4j Browser.
Downstream consumers are the risk engine, the agent's tools and the UI API. All of them read
the warehouse, not Neo4j.

## Alternatives considered
- **Neo4j as the only store.** Cleaning and entity resolution are set operations over
  ~74 000 sanctions rows, which is natural in SQL and awkward in Cypher. And every unit test
  would need a database server.
- **Postgres as the warehouse.** It works, but it is a server to run. DuckDB is a file that
  tests create in memory in milliseconds.

## Consequences
- Unit tests never need a server. Neo4j is exercised by opt-in integration tests (`TESTSG*`
  nodes only).
- Traversals in Python (risk propagation, `find_connection`, neighbourhoods) run over edges
  loaded from DuckDB. This is fine at thousands of edges. At millions, the risk engine and the
  path tools would move to Neo4j / GDS, and this record would be superseded.
- DuckDB allows one writer process. Readers must open read-only and must not overlap a
  pipeline write. 0011 shows how the deployment handles that.
