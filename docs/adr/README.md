# Architecture decision records

Short records of the decisions that shape SupplyGraph: the context, what was decided, what
was rejected and why, and what it costs. They follow Michael Nygard's format. A record is never
edited to reverse a decision; a new record supersedes it.

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-duckdb-warehouse-neo4j-mirror.md) | DuckDB is the source of truth; Neo4j mirrors it | Accepted |
| [0003](0003-ontology-link-types-encode-certainty.md) | The ontology is the contract; link types encode certainty | Accepted |
| [0004](0004-idempotent-pipeline-one-implementation.md) | Idempotent full-refresh steps, one implementation, blocking quality gates | Accepted |
| [0005](0005-entity-resolution-identifiers-first.md) | Entity resolution: identifiers first, dev-tuned thresholds, LLM only for the review band | Accepted |
| [0006](0006-supplier-extraction-measured-against-gold.md) | Supplier extraction is measured against a hand-labelled gold set | Accepted |
| [0007](0007-explainable-risk-propagation-no-gnn.md) | Explainable rule-based risk propagation; no GNN at this scale | Accepted |
| [0008](0008-agent-tools-grounded-citations.md) | The agent retrieves through typed tools and must cite what it retrieved | Accepted |
| [0009](0009-proposal-only-actions-append-only-log.md) | The agent only proposes actions; decisions go to an append-only log | Accepted |
| [0010](0010-api-write-protection-modes.md) | API write protection: open on localhost, token or read-only elsewhere | Accepted |
| [0011](0011-kubernetes-single-image-cronjob-publish.md) | Kubernetes: one image, CronJob builds and atomically publishes the warehouse | Accepted |
| [0012](0012-observability-freshness-first.md) | Observability: data freshness first, then traffic | Accepted |
| [0013](0013-eval-tracking-prompt-versions-llm-judge.md) | Eval runs tracked in MLflow; prompts versioned and locked; LLM judge advisory | Accepted |
| [0014](0014-pgvector-candidate-generation-opt-in.md) | pgvector candidate generation: measured, kept opt-in | Accepted |
| [0015](0015-celery-background-jobs.md) | Celery + Redis for on-demand background jobs; Dagster stays the scheduler | Accepted |
