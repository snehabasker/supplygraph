"""Dagster software-defined assets for the SupplyGraph pipeline.

    gleif_raw ──┐
                ├─► warehouse ──[quality_gate: blocking]──► neo4j_graph
    sanctions_raw┘

Run locally from the repo root:  dagster dev   (UI at http://localhost:3000)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetSelection,
    Backoff,
    ConfigurableResource,
    Definitions,
    MaterializeResult,
    MetadataValue,
    RetryPolicy,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)

from supplygraph import pipeline, quality
from supplygraph.config import Settings, load_settings

NETWORK_RETRY = RetryPolicy(max_retries=2, delay=60, backoff=Backoff.EXPONENTIAL)


class SupplyGraphResource(ConfigurableResource):
    """Pipeline settings. Defaults come from the environment / .env; fields override them
    (tests point `data_dir` at a temp folder and `neo4j_uri` at a dead port)."""

    seeds_path: str = "seeds/companies.csv"
    children_per_seed: int = 25
    data_dir: str | None = None
    neo4j_uri: str | None = None

    def settings(self) -> Settings:
        settings = load_settings()
        if self.data_dir:
            settings = replace(settings, data_dir=Path(self.data_dir))
        if self.neo4j_uri:
            settings = replace(settings, neo4j_uri=self.neo4j_uri)
        return settings


@asset(group_name="ingestion", kinds={"python"}, retry_policy=NETWORK_RETRY,
       description="Seed companies + parents/subsidiaries from the GLEIF API (raw JSONL).")
def gleif_raw(sg: SupplyGraphResource) -> MaterializeResult:
    result = pipeline.ingest_gleif(sg.settings(), Path(sg.seeds_path), sg.children_per_seed)
    return MaterializeResult(metadata={
        "entities": len(result.records),
        "ownership_links": len(result.relationships),
        "missing_seeds": MetadataValue.json(result.missing_seeds),
    })


@asset(group_name="ingestion", kinds={"python"}, retry_policy=NETWORK_RETRY,
       description="OpenSanctions consolidated sanctions list (raw CSV).")
def sanctions_raw(sg: SupplyGraphResource) -> MaterializeResult:
    path = pipeline.ingest_sanctions(sg.settings())
    return MaterializeResult(metadata={
        "path": MetadataValue.path(str(path)),
        "size_mb": round(path.stat().st_size / 1e6, 1),
    })


@asset(deps=[gleif_raw, sanctions_raw], group_name="warehouse", kinds={"duckdb"},
       description="Companies, ownership, sanctions and LEI matches in DuckDB.")
def warehouse(sg: SupplyGraphResource) -> MaterializeResult:
    return MaterializeResult(metadata=dict(pipeline.build(sg.settings())))


@asset_check(asset=warehouse, blocking=True,
             description="All error-level data quality checks pass; blocks the graph load if not.")
def quality_gate(sg: SupplyGraphResource) -> AssetCheckResult:
    results = pipeline.check(sg.settings())
    failing = [r for r in results if r.status == "fail"]
    return AssetCheckResult(
        passed=not quality.has_blocking_failures(results),
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "checks_run": sum(r.status != "skipped" for r in results),
            "failing": MetadataValue.json(
                [{"name": r.name, "severity": r.severity, "failures": r.failures}
                 for r in failing]
            ),
        },
    )


@asset(deps=[warehouse], group_name="graph", kinds={"neo4j"}, retry_policy=NETWORK_RETRY,
       description="Ontology-shaped knowledge graph in Neo4j (idempotent MERGE).")
def neo4j_graph(sg: SupplyGraphResource) -> MaterializeResult:
    return MaterializeResult(metadata=dict(pipeline.load_graph(sg.settings())))


ALL_ASSETS = [gleif_raw, sanctions_raw, warehouse, neo4j_graph]

full_refresh_job = define_asset_job(
    "full_refresh", selection=AssetSelection.all(),
    description="Re-crawl GLEIF, refresh sanctions, rebuild warehouse and graph.",
)
sanctions_refresh_job = define_asset_job(
    "sanctions_refresh", selection=AssetSelection.assets(sanctions_raw).downstream(),
    description="Sanctions change daily, ownership rarely: refresh only what depends on them.",
)

defs = Definitions(
    assets=ALL_ASSETS,
    asset_checks=[quality_gate],
    jobs=[full_refresh_job, sanctions_refresh_job],
    schedules=[
        ScheduleDefinition(job=sanctions_refresh_job, cron_schedule="0 6 * * *",
                           execution_timezone="UTC"),
        ScheduleDefinition(job=full_refresh_job, cron_schedule="0 5 * * 1",
                           execution_timezone="UTC"),
    ],
    resources={"sg": SupplyGraphResource()},
)
