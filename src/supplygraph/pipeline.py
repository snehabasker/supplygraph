"""Pipeline steps shared by the CLI and Dagster assets: one implementation, two entry points."""

from __future__ import annotations

import json
from datetime import UTC, datetime
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
    # The seeds are the companies the user cares about: they go on the news watchlist.
    (settings.gleif_dir / "seeds.json").write_text(json.dumps(leis), encoding="utf-8")
    return result


def ingest_filings(settings: Settings, filers_csv: Path) -> list:
    from supplygraph.ingest import sec

    filers = sec.read_filers(filers_csv)
    with sec.SecClient(sec.check_user_agent(settings.sec_user_agent)) as client:
        return sec.ingest_filings(client, filers, settings.sec_dir)


def make_extractor(method: str, aliases: dict[str, str], con=None):
    from supplygraph.extract import suppliers

    if method == suppliers.RULES:
        return suppliers.RuleExtractor(aliases)
    if method == suppliers.LLM:
        inner = suppliers.ClaudeExtractor(aliases)
        return suppliers.CachedClaudeExtractor(inner, con) if con is not None else inner
    raise ValueError(f"unknown extraction method {method!r}; use one of {suppliers.METHODS}")


def extract_suppliers(settings: Settings, method: str = "RULES") -> dict[str, int]:
    from supplygraph.extract import supply_chain
    from supplygraph.ingest import sec

    refs = sec.read_manifest(settings.sec_dir)
    if not refs:
        raise FileNotFoundError(f"No filings in {settings.sec_dir}; run `ingest-filings` first")
    texts = supply_chain.load_texts(settings.sec_dir, refs)
    aliases, _ = supply_chain.build_passages(refs, texts)
    with (
        warehouse.connect(settings.warehouse_path) as con,
        gleif.GleifClient(settings.gleif_base_url, settings.user_agent) as client,
    ):
        extractor = make_extractor(method.upper(), aliases, con)
        return supply_chain.run(settings.sec_dir, settings.supplier_gleif_dir, refs,
                                extractor, client)


def build_extraction_eval_set(settings: Settings, eval_dir: Path) -> int:
    from supplygraph.extract import supply_chain
    from supplygraph.extract.evaluate import write_eval_set
    from supplygraph.ingest import sec

    refs = sec.read_manifest(settings.sec_dir)
    texts = supply_chain.load_texts(settings.sec_dir, refs)
    aliases, passages = supply_chain.build_passages(refs, texts)
    contexts = {ref.cik: supply_chain.filer_context(ref) for ref in refs}
    return write_eval_set(eval_dir, aliases, contexts, passages)


def evaluate_extraction(settings: Settings, eval_dir: Path, use_llm: bool = False) -> dict:
    from supplygraph.extract import suppliers
    from supplygraph.extract.evaluate import evaluate, load_eval_set

    aliases, _, _ = load_eval_set(eval_dir)
    if not use_llm:
        return evaluate(eval_dir, {"rules": suppliers.RuleExtractor(aliases)})
    with warehouse.connect(settings.warehouse_path) as con:
        return evaluate(eval_dir, {
            "rules": suppliers.RuleExtractor(aliases),
            "llm": make_extractor(suppliers.LLM, aliases, con),
        })


def ingest_news(settings: Settings, hours: float = 24, end: datetime | None = None) -> dict:
    from supplygraph.ingest import gdelt

    with gdelt.GdeltClient(settings.user_agent) as client:
        return gdelt.ingest_news(client, end or datetime.now(UTC), hours, settings.gdelt_dir)


def ingest_sanctions(settings: Settings) -> Path:
    return opensanctions.download(
        settings.opensanctions_url, settings.sanctions_csv, settings.user_agent
    )


def build(settings: Settings) -> dict[str, int]:
    with warehouse.connect(settings.warehouse_path) as con:
        return warehouse.build_all(con, settings.gleif_dir, settings.sanctions_csv,
                                   settings.sec_dir, settings.gdelt_dir)


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


def score_risk(settings: Settings) -> dict[str, int]:
    from supplygraph.risk import engine

    cfg = engine.RiskConfig()
    with warehouse.connect(settings.warehouse_path) as con:
        return engine.write_scores(con, engine.score(con, cfg), cfg)


def find_company(con, query: str) -> tuple[str, str]:
    """An LEI, or a name / short name ("TSMC") -> (lei, legal name). Raises if not unique."""
    from supplygraph.er.normalize import normalize
    from supplygraph.ingest.gleif import is_valid_lei

    names = dict(con.execute("SELECT lei, legal_name FROM companies").fetchall())
    if is_valid_lei(query.strip().upper()) and query.strip().upper() in names:
        lei = query.strip().upper()
        return lei, names[lei]
    target = normalize(query)
    matches = set()
    for lei, legal_name, other_names in con.execute(
        "SELECT lei, legal_name, other_names FROM companies"
    ).fetchall():
        if any(normalize(n) == target for n in [legal_name, *(other_names or "").split(";")]):
            matches.add(lei)
    if not matches and warehouse.table_exists(con, "supplier_mentions"):
        # Short names from filings: "TSMC", "Samsung".
        for lei, written in con.execute(
            "SELECT DISTINCT supplier_lei, name_as_written FROM supplier_mentions "
            "WHERE supplier_lei IS NOT NULL"
        ).fetchall():
            if normalize(written) == target:
                matches.add(lei)
    if len(matches) != 1:
        found = ", ".join(f"{names[m]} ({m})" for m in sorted(matches)) or "none"
        raise ValueError(f"{query!r} must match exactly one company; found: {found}")
    lei = matches.pop()
    return lei, names[lei]


def risk_scenario(settings: Settings, company: str, severity: float = 1.0) -> dict:
    from supplygraph.risk import engine

    with warehouse.connect(settings.warehouse_path) as con:
        lei, name = find_company(con, company)
        return {"lei": lei, "name": name, "severity": severity,
                "impacted": engine.scenario(con, lei, severity)}


def evaluate_risk(settings: Settings) -> dict:
    from supplygraph.risk import engine, linkpred

    with warehouse.connect(settings.warehouse_path) as con:
        return {"sensitivity": engine.sensitivity(con),
                "link_prediction": linkpred.leave_one_out(linkpred.load_links(con))}


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
