"""Pipeline steps shared by the CLI and Dagster assets: one implementation, two entry points."""

from __future__ import annotations

from pathlib import Path

from supplygraph import graph, quality, warehouse
from supplygraph.config import Settings
from supplygraph.ingest import gleif, opensanctions


def ingest_gleif(
    settings: Settings, seeds: Path, children: int = 25, max_seeds: int | None = None
) -> gleif.CrawlResult:
    leis = gleif.read_seed_leis(seeds)[:max_seeds]
    with gleif.GleifClient(settings.gleif_base_url, settings.user_agent) as client:
        result = gleif.crawl(client, leis, children_limit=children)
    gleif.write_crawl(result, settings.gleif_dir)
    return result


def ingest_sanctions(settings: Settings) -> Path:
    return opensanctions.download(
        settings.opensanctions_url, settings.sanctions_csv, settings.user_agent
    )


def build(settings: Settings) -> dict[str, int]:
    with warehouse.connect(settings.warehouse_path) as con:
        return warehouse.build_all(con, settings.gleif_dir, settings.sanctions_csv)


def resolve_entities(settings: Settings, use_llm: bool = False) -> dict[str, int]:
    from supplygraph.er.resolve import resolve

    adjudicator = None
    if use_llm:
        from supplygraph.er.adjudicate import ClaudeAdjudicator

        adjudicator = ClaudeAdjudicator()
    with warehouse.connect(settings.warehouse_path) as con:
        return resolve(con, adjudicator)


def build_eval_set(settings: Settings, eval_dir: Path) -> dict[str, int]:
    from supplygraph.er.evaluate import build_eval_set as build

    with (
        warehouse.connect(settings.warehouse_path) as con,
        gleif.GleifClient(settings.gleif_base_url, settings.user_agent) as client,
    ):
        return build(con, client, eval_dir)


def evaluate_entities(settings: Settings, eval_dir: Path, use_llm: bool = False) -> dict:
    from supplygraph.er.evaluate import evaluate

    adjudicator = None
    if use_llm:
        from supplygraph.er.adjudicate import ClaudeAdjudicator

        adjudicator = ClaudeAdjudicator()
    with warehouse.connect(settings.warehouse_path) as con:
        return evaluate(con, eval_dir, adjudicator=adjudicator)


def check(settings: Settings) -> list[quality.CheckResult]:
    with warehouse.connect(settings.warehouse_path) as con:
        return quality.run_checks(con)


def load_graph(settings: Settings, batch_size: int = 1000) -> dict[str, int]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        driver.verify_connectivity()
        with warehouse.connect(settings.warehouse_path) as con:
            return graph.load_graph(con, driver, batch_size=batch_size)
    finally:
        driver.close()
