"""Command-line entry point: `supplygraph <command>`."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

from supplygraph import pipeline, quality
from supplygraph.config import load_settings

app = typer.Typer(help="SupplyGraph: supplier risk knowledge graph pipeline.", no_args_is_help=True)

DEFAULT_SEEDS = Path("seeds/companies.csv")
DEFAULT_EVAL_DIR = Path("eval")


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
    result = pipeline.ingest_gleif(settings, seeds, children, max_seeds)
    typer.echo(f"GLEIF: {len(result.records)} entities, {len(result.relationships)} ownership "
               f"links -> {settings.gleif_dir}")
    if result.missing_seeds:
        typer.echo(f"WARNING: seeds not found: {result.missing_seeds}", err=True)


@app.command("ingest-sanctions")
def ingest_sanctions() -> None:
    """Download the OpenSanctions consolidated sanctions list."""
    typer.echo(f"OpenSanctions -> {pipeline.ingest_sanctions(load_settings())}")


@app.command()
def build() -> None:
    """Build warehouse tables from raw files and link companies to sanctions."""
    for table, n in pipeline.build(load_settings()).items():
        typer.echo(f"{table:<26}{n:>8}")


@app.command()
def resolve(
    llm: bool = typer.Option(False, help="Ask Claude to adjudicate the review band."),
) -> None:
    """Fuzzy-match companies to sanctions records (after `build`)."""
    for key, n in pipeline.resolve_entities(load_settings(), use_llm=llm).items():
        typer.echo(f"{key:<18}{n:>8}")


@app.command("build-eval-set")
def build_eval_set(eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR)) -> None:
    """Rebuild the entity-resolution gold set from live GLEIF + current sanctions data."""
    counts = pipeline.build_eval_set(load_settings(), eval_dir)
    typer.echo(f"Eval set -> {eval_dir}: {counts}")


@app.command("evaluate-er")
def evaluate_er(
    eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR),
    llm: bool = typer.Option(False, help="Also measure the hybrid matcher + Claude system."),
    out: Path = typer.Option(Path("data/er_eval_report.json"), help="Full JSON report."),
) -> None:
    """Measure entity-resolution precision/recall on the gold set (dev-tuned, test-reported)."""
    report = pipeline.evaluate_entities(load_settings(), eval_dir, use_llm=llm)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    t = report["chosen_thresholds_on_dev"]
    typer.echo(f"Companies: {report['companies']}   blocking recall: {report['blocking_recall']}")
    typer.echo(f"Thresholds (chosen on dev): auto_match={t['auto_match']} review={t['review']}")
    typer.echo(f"\n{'TEST':<34}{'precision':>10}{'recall':>8}{'F1':>7}{'false alarm':>13}")
    rows = {**{k: v for k, v in report["test"].items() if "precision" in v},
            **{f"ablation: {k}": v for k, v in report["test_ablations_at_auto_threshold"].items()}}
    for name, m in rows.items():
        typer.echo(f"{name:<34}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}"
                   f"{m['false_alarm_rate']:>13.3%}")
    queue = report["test"]["review_queue"]
    typer.echo(f"\nReview queue (test): {queue['size']} items, "
               f"{queue['true_matches_in_queue']} true matches")
    typer.echo(f"Full report -> {out}")


@app.command()
def check() -> None:
    """Run data quality checks. Exits with code 1 if any error-level check fails."""
    results = pipeline.check(load_settings())
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
    for label, n in pipeline.load_graph(load_settings(), batch_size).items():
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
    resolve(llm=False)
    check()
    if not skip_graph:
        load_graph(batch_size=1000)


if __name__ == "__main__":
    app()
