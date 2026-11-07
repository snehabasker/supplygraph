"""Command-line entry point: `supplygraph <command>`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from supplygraph import graph, quality, warehouse
from supplygraph.config import load_settings
from supplygraph.ingest import gleif, opensanctions

app = typer.Typer(help="SupplyGraph: supplier risk knowledge graph pipeline.", no_args_is_help=True)

DEFAULT_SEEDS = Path("seeds/companies.csv")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    # Windows consoles default to cp1252; company names include Cyrillic, CJK, etc.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)  # "constraint already exists" noise


@app.command("ingest-gleif")
def ingest_gleif(
    seeds: Path = typer.Option(DEFAULT_SEEDS, help="CSV with an `lei` column."),
    children: int = typer.Option(25, min=0, help="Max direct subsidiaries fetched per seed."),
    max_seeds: int | None = typer.Option(None, min=1, help="Only crawl the first N seeds."),
) -> None:
    """Fetch seed companies and their ownership network from GLEIF."""
    settings = load_settings()
    leis = gleif.read_seed_leis(seeds)[:max_seeds]
    with gleif.GleifClient(settings.gleif_base_url, settings.user_agent) as client:
        result = gleif.crawl(client, leis, children_limit=children)
    gleif.write_crawl(result, settings.gleif_dir)
    typer.echo(f"GLEIF: {len(result.records)} entities, {len(result.relationships)} ownership "
               f"links -> {settings.gleif_dir}")
    if result.missing_seeds:
        typer.echo(f"WARNING: seeds not found: {result.missing_seeds}", err=True)


@app.command("ingest-sanctions")
def ingest_sanctions() -> None:
    """Download the OpenSanctions consolidated sanctions list."""
    settings = load_settings()
    path = opensanctions.download(
        settings.opensanctions_url, settings.sanctions_csv, settings.user_agent
    )
    typer.echo(f"OpenSanctions -> {path}")


@app.command()
def build() -> None:
    """Build warehouse tables from raw files and link companies to sanctions."""
    settings = load_settings()
    with warehouse.connect(settings.warehouse_path) as con:
        counts = warehouse.build_all(con, settings.gleif_dir, settings.sanctions_csv)
    for table, n in counts.items():
        typer.echo(f"{table:<26}{n:>8}")


@app.command()
def check() -> None:
    """Run data quality checks. Exits with code 1 if any error-level check fails."""
    settings = load_settings()
    with warehouse.connect(settings.warehouse_path) as con:
        results = quality.run_checks(con)
    for r in results:
        marker = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[r.status]
        typer.echo(f"[{marker}] {r.severity:<5} {r.name:<28} failures={r.failures:<6} "
                   f"{r.description}")
    if quality.has_blocking_failures(results):
        typer.echo("Blocking data quality failures; refusing to continue.", err=True)
        raise typer.Exit(code=1)


@app.command("load-graph")
def load_graph(batch_size: int = typer.Option(1000, min=1)) -> None:
    """Load the warehouse into Neo4j (idempotent MERGE)."""
    from neo4j import GraphDatabase

    settings = load_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        driver.verify_connectivity()
        with warehouse.connect(settings.warehouse_path) as con:
            counts = graph.load_graph(con, driver, batch_size=batch_size)
    finally:
        driver.close()
    for label, n in counts.items():
        typer.echo(f"{label:<22}{n:>8}")


@app.command("run-all")
def run_all(
    seeds: Path = typer.Option(DEFAULT_SEEDS),
    children: int = typer.Option(25, min=0),
    skip_graph: bool = typer.Option(False, help="Stop after quality checks."),
) -> None:
    """Ingest -> build -> check -> load graph. Stops if quality checks fail."""
    ingest_gleif(seeds=seeds, children=children, max_seeds=None)
    ingest_sanctions()
    build()
    check()
    if not skip_graph:
        load_graph(batch_size=1000)


if __name__ == "__main__":
    app()
