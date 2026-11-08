"""Dagster software-defined assets for the SupplyGraph pipeline.

    gleif_raw ─────────────────────┐
    sanctions_raw ─────────────────┤
    sec_filings ─► supplier_extraction ─┼─► warehouse ─[quality_gate]─► entity_resolution
    news_raw ──────────────────────┘                                   ─[match_gate]─► neo4j_graph

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
    filers_path: str = "seeds/sec_filers.csv"
    children_per_seed: int = 25
    use_llm: bool = False  # Claude adjudicates the ER review band (needs Anthropic credentials)
    extraction_method: str = "rules"  # "llm" -> SUPPLIES links (needs Anthropic credentials)
    news_hours: float = 24
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


@asset(group_name="ingestion", kinds={"python"}, retry_policy=NETWORK_RETRY,
       description="Latest 10-K / 20-F of each SEC filer (raw HTML + manifest). Skipped, "
                   "not failed, when SEC_USER_AGENT is not configured.")
def sec_filings(sg: SupplyGraphResource) -> MaterializeResult:
    settings = sg.settings()
    if not settings.sec_user_agent:
        return MaterializeResult(metadata={"skipped": "SEC_USER_AGENT not set"})
    refs = pipeline.ingest_filings(settings, Path(sg.filers_path))
    return MaterializeResult(metadata={
        "filings": len(refs),
        "forms": MetadataValue.json({r.entity_name: f"{r.form} {r.filing_date}" for r in refs}),
    })


@asset(deps=[sec_filings], group_name="ingestion", kinds={"python"}, retry_policy=NETWORK_RETRY,
       description="Supplier mentions extracted from filings, resolved to LEIs via GLEIF.")
def supplier_extraction(sg: SupplyGraphResource) -> MaterializeResult:
    settings = sg.settings()
    if not (settings.sec_dir / "filings.jsonl").exists():
        return MaterializeResult(metadata={"skipped": "no filings ingested"})
    return MaterializeResult(metadata=dict(
        pipeline.extract_suppliers(settings, sg.extraction_method)))


@asset(group_name="ingestion", kinds={"python"}, retry_policy=NETWORK_RETRY,
       description="GDELT GKG articles tagged with disruption themes (raw JSONL).")
def news_raw(sg: SupplyGraphResource) -> MaterializeResult:
    return MaterializeResult(metadata=dict(pipeline.ingest_news(sg.settings(), sg.news_hours)))


@asset(deps=[gleif_raw, sanctions_raw, supplier_extraction, news_raw], group_name="warehouse",
       kinds={"duckdb"},
       description="Companies, ownership, sanctions, LEI matches, supplier links and news.")
def warehouse(sg: SupplyGraphResource) -> MaterializeResult:
    return MaterializeResult(metadata=dict(pipeline.build(sg.settings())))


def _gate(sg: SupplyGraphResource) -> AssetCheckResult:
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


@asset_check(asset=warehouse, blocking=True,
             description="All error-level data quality checks pass before entity resolution.")
def quality_gate(sg: SupplyGraphResource) -> AssetCheckResult:
    return _gate(sg)


@asset(deps=[warehouse], group_name="warehouse", kinds={"python", "duckdb"},
       description="Fuzzy company<->sanctions matching; optional Claude adjudication; "
                   "borderline cases go to a review queue.")
def entity_resolution(sg: SupplyGraphResource) -> MaterializeResult:
    return MaterializeResult(metadata=dict(pipeline.resolve_entities(sg.settings(), sg.use_llm)))


@asset_check(asset=entity_resolution, blocking=True,
             description="Match invariants hold (methods, confidence rule, evidence, references).")
def match_gate(sg: SupplyGraphResource) -> AssetCheckResult:
    return _gate(sg)


@asset(deps=[entity_resolution], group_name="graph", kinds={"neo4j"}, retry_policy=NETWORK_RETRY,
       description="Ontology-shaped knowledge graph in Neo4j (idempotent MERGE).")
def neo4j_graph(sg: SupplyGraphResource) -> MaterializeResult:
    return MaterializeResult(metadata=dict(pipeline.load_graph(sg.settings())))


ALL_ASSETS = [gleif_raw, sanctions_raw, sec_filings, supplier_extraction, news_raw, warehouse,
              entity_resolution, neo4j_graph]
ALL_CHECKS = [quality_gate, match_gate]

full_refresh_job = define_asset_job(
    "full_refresh", selection=AssetSelection.all(),
    description="Re-crawl GLEIF, refresh sanctions, rebuild warehouse and graph.",
)
sanctions_refresh_job = define_asset_job(
    "sanctions_refresh", selection=AssetSelection.assets(sanctions_raw).downstream(),
    description="Sanctions change daily, ownership rarely: refresh only what depends on them.",
)

news_refresh_job = define_asset_job(
    "news_refresh", selection=AssetSelection.assets(news_raw).downstream(),
    description="Disruption news moves fastest: refresh it and everything downstream.",
)

defs = Definitions(
    assets=ALL_ASSETS,
    asset_checks=ALL_CHECKS,
    jobs=[full_refresh_job, sanctions_refresh_job, news_refresh_job],
    schedules=[
        ScheduleDefinition(job=news_refresh_job, cron_schedule="30 */6 * * *",
                           execution_timezone="UTC"),
        ScheduleDefinition(job=sanctions_refresh_job, cron_schedule="0 6 * * *",
                           execution_timezone="UTC"),
        ScheduleDefinition(job=full_refresh_job, cron_schedule="0 5 * * 1",
                           execution_timezone="UTC"),
    ],
    resources={"sg": SupplyGraphResource()},
)
