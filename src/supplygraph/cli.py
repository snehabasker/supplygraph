"""Command-line entry point: `supplygraph <command>`."""

from __future__ import annotations

import getpass
import json
import sys
from pathlib import Path

import typer

from supplygraph import logs, pipeline, quality
from supplygraph.config import load_settings

app = typer.Typer(help="SupplyGraph: supplier risk knowledge graph pipeline.", no_args_is_help=True)

DEFAULT_SEEDS = Path("seeds/companies.csv")
DEFAULT_FILERS = Path("seeds/sec_filers.csv")
DEFAULT_EVAL_DIR = Path("eval")
TRACK_HELP = "Also log params, metrics and the report to MLflow (MLFLOW_TRACKING_URI)."


def _track(name: str, report: dict, params: dict | None = None,
           series: dict[str, list[tuple[int, float]]] | None = None) -> None:
    """Log an evaluation run to MLflow (`pip install -e .[tracking]`)."""
    import os

    from supplygraph import tracking

    settings = load_settings()
    try:
        run_id = tracking.log_eval(
            name, report, params=params, series=series,
            tracking_uri=tracking.default_tracking_uri(settings.data_dir),
            # With a tracking server, the server decides where artifacts go.
            artifact_root=None if os.getenv("MLFLOW_TRACKING_URI")
            else settings.data_dir / "mlartifacts")
    except ModuleNotFoundError as exc:
        typer.echo(f"MLflow is not installed ({exc}); pip install -e .[tracking]", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"MLflow run {run_id} -> experiment {tracking.EXPERIMENT_PREFIX}/{name}")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    # Windows consoles default to cp1252; company names include Cyrillic, CJK, etc.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logs.configure(verbose)


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


@app.command("ingest-filings")
def ingest_filings(
    filers: Path = typer.Option(DEFAULT_FILERS, help="CSV with lei, cik, name columns."),
) -> None:
    """Download each filer's latest 10-K / 20-F from SEC EDGAR (needs SEC_USER_AGENT)."""
    settings = load_settings()
    refs = pipeline.ingest_filings(settings, filers)
    for ref in refs:
        typer.echo(f"{ref.form:<6}{ref.filing_date}  {ref.entity_name}")
    typer.echo(f"SEC: {len(refs)} filings -> {settings.sec_dir}")


@app.command("extract-suppliers")
def extract_suppliers(
    method: str = typer.Option("rules", help="rules (baseline, -> POSSIBLE_SUPPLIER) or "
                                             "llm (Claude, -> SUPPLIES)."),
) -> None:
    """Extract supplier relationships from the ingested filings and resolve them to LEIs."""
    counts = pipeline.extract_suppliers(load_settings(), method)
    for key, n in counts.items():
        typer.echo(f"{key:<20}{n:>8}")
    if method.lower() == "rules":
        typer.echo("Rule-based links load as POSSIBLE_SUPPLIER (0.56 precision on the gold "
                   "set); use --method llm for SUPPLIES.")


@app.command("ingest-news")
def ingest_news(
    hours: float = typer.Option(24, min=0.25, help="How far back to read GDELT (15-min files)."),
) -> None:
    """Fetch GDELT GKG files and keep articles tagged with disruption themes."""
    counts = pipeline.ingest_news(load_settings(), hours)
    typer.echo(f"GDELT: {counts}")


@app.command("build-extraction-eval-set")
def build_extraction_eval_set(
    eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR / "extraction"),
) -> None:
    """Freeze candidate passages from the ingested filings (gold.csv is hand-labelled)."""
    n = pipeline.build_extraction_eval_set(load_settings(), eval_dir)
    typer.echo(f"{n} passages -> {eval_dir}  (re-check gold.csv against them)")


@app.command("evaluate-extraction")
def evaluate_extraction(
    eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR / "extraction"),
    llm: bool = typer.Option(False, help="Also measure the Claude extractor."),
    out: Path = typer.Option(Path("data/extraction_eval_report.json")),
    track: bool = typer.Option(False, help=TRACK_HELP),
) -> None:
    """Supplier-extraction precision/recall on the hand-labelled gold set."""
    report = pipeline.evaluate_extraction(load_settings(), eval_dir, use_llm=llm)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"{report['passages']} passages from {report['filers']} filings; "
               f"{report['gold_suppliers']} gold supplier relationships")
    typer.echo(f"\n{'system':<10}{'precision':>10}{'recall':>8}{'F1':>7}{'TP':>5}{'FP':>5}"
               f"{'FN':>5}{'failed':>8}")
    for name, m in report["systems"].items():
        typer.echo(f"{name:<10}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}"
                   f"{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}{m['failed_passages']:>8}")
    typer.echo(f"Full report (false positives, misses) -> {out}")
    if track:
        import os

        from supplygraph.extract.suppliers import DEFAULT_MODEL, PROMPT_VERSION

        _track("extraction", report, {
            "systems": sorted(report["systems"]), "prompt.extract.suppliers": PROMPT_VERSION,
            "model": os.getenv("SUPPLYGRAPH_LLM_MODEL", DEFAULT_MODEL) if llm else None})


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
    vectors: bool = typer.Option(False, help="Add pgvector nearest-neighbour candidates "
                                             "(PGVECTOR_DSN) and compare with token blocking."),
    out: Path = typer.Option(Path("data/er_eval_report.json"), help="Full JSON report."),
    track: bool = typer.Option(False, help=TRACK_HELP),
) -> None:
    """Measure entity-resolution precision/recall on the gold set (dev-tuned, test-reported)."""
    report = pipeline.evaluate_entities(load_settings(), eval_dir, use_llm=llm, vectors=vectors)
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
    if "blocking_comparison" in report:
        typer.echo(f"\n{'BLOCKING (test, own dev thresholds)':<38}{'blocking recall':>16}"
                   f"{'precision':>10}{'recall':>8}{'false alarm':>13}")
        for name, b in report["blocking_comparison"].items():
            m = b["test_auto_match"]
            typer.echo(f"{name:<38}{b['blocking_recall']:>16.3f}{m['precision']:>10.3f}"
                       f"{m['recall']:>8.3f}{m['false_alarm_rate']:>13.3%}")
    typer.echo(f"Full report -> {out}")
    if track:
        from dataclasses import asdict

        from supplygraph.er.matcher import MatcherConfig

        params = {**{f"matcher.{k}": v for k, v in asdict(MatcherConfig()).items()},
                  "blocking": report["blocking"]["method"],
                  "matcher.vector_k": report["blocking"]["vector_k"],
                  **{f"threshold.{k}": v for k, v in t.items()}}
        if llm:
            from supplygraph.er.adjudicate import PROMPT_VERSION

            params.update({"model": report["test"]["hybrid_with_llm"]["model"],
                           "prompt.er.adjudicate": PROMPT_VERSION})
        sweep = report["dev_sweep"]
        _track("entity_resolution", report, params, series={
            f"dev_sweep.{k}": [(round(m["threshold"] * 100), m[k]) for m in sweep]
            for k in ("precision", "recall", "f1")})


@app.command("score-risk")
def score_risk() -> None:
    """Propagate risk over ownership and supplier links (after `resolve`)."""
    for key, n in pipeline.score_risk(load_settings()).items():
        typer.echo(f"{key:<20}{n:>8}")


@app.command()
def scenario(
    company: str = typer.Argument(..., help="LEI, legal name or short name, e.g. TSMC."),
    severity: float = typer.Option(1.0, min=0.0, max=1.0),
    top: int = typer.Option(15, min=1),
) -> None:
    """What if COMPANY is disrupted? Companies whose risk rises, and through which links."""
    result = pipeline.risk_scenario(load_settings(), company, severity)
    impacted = result["impacted"]
    typer.echo(f"Disruption of {result['name']} ({result['lei']}), severity {severity}: "
               f"{len(impacted)} companies affected")
    for r in impacted[:top]:
        via = " -> ".join(r["path"])
        typer.echo(f"  +{r['delta']:.3f}  {r['name']}\n         via {via} "
                   f"[{', '.join(r['relations'])}]")


@app.command()
def twin(
    company: str = typer.Argument(..., help="LEI, legal name or short name, e.g. TSMC."),
    severity: float = typer.Option(1.0, min=0.0, max=1.0, help="Share of capacity lost."),
    duration: int = typer.Option(14, min=0, help="Days at reduced capacity."),
    recovery: int = typer.Option(7, min=0, help="Days to ramp back to full capacity."),
    inventory: float = typer.Option(10.0, min=0.0, help="Days of input stock per company."),
    days: int = typer.Option(60, min=1, help="Simulation horizon."),
    top: int = typer.Option(15, min=1),
    out: Path | None = typer.Option(None, help="Write the full timeline as JSON."),
) -> None:
    """Digital twin: simulate a disruption at COMPANY day by day, with inventory buffers."""
    r = pipeline.twin_scenario(load_settings(), company, severity, duration, recovery,
                               inventory, days)
    typer.echo(f"Twin: {r['name']} loses {severity:.0%} capacity for {duration}d "
               f"(+{recovery}d recovery), {inventory:g}d inventory, {days}d horizon: "
               f"{len(r['impacted'])} downstream companies hit")
    for i in r["impacted"][:top]:
        so = f"stockout day {i['stockout_day']}" if i["stockout_day"] is not None else "no stockout"
        typer.echo(f"  day {i['first_impact_day']:>3}  min {i['min_capacity']:.0%}  "
                   f"{i['days_impacted']:>3}d hit  {so:<16} {i['name']}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"Timeline written to {out}")


@app.command("evaluate-news-clusters")
def evaluate_news_clusters(
    eval_dir: Path = typer.Option(Path("eval/news")),
    out: Path = typer.Option(Path("data/news_cluster_eval_report.json")),
    track: bool = typer.Option(False, help=TRACK_HELP),
) -> None:
    """Story-clustering precision/recall on labelled headline pairs (dev-tuned, test-reported)."""
    from supplygraph import news_cluster

    report = news_cluster.evaluate(eval_dir / "events.jsonl", eval_dir / "gold_pairs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lab = report["labelled"]
    typer.echo(f"{report['events']} events; labelled pairs dev={lab['dev']['pairs']} "
               f"test={lab['test']['pairs']} (stratified sample, not the natural mix)")
    typer.echo(f"{'system':<24}{'thr':>6}{'P(test)':>9}{'R(test)':>9}{'F1':>7}")
    for name, m in report["systems"].items():
        t = m["test"]
        thr = "-" if m["threshold"] is None else f"{m['threshold']:.2f}"
        typer.echo(f"{name:<24}{thr:>6}{t['precision']:>9.3f}{t['recall']:>9.3f}{t['f1']:>7.3f}")
    errs = report["test_errors"]
    typer.echo(f"\nTest false merges ({len(errs['false_merges'])}):")
    for a, b in errs["false_merges"]:
        typer.echo(f"  {a[:70]}  <>  {b[:70]}")
    typer.echo(f"Full report -> {out}")
    if track:
        _track("news_clusters", report, {f"threshold.{name}": m["threshold"]
                                         for name, m in report["systems"].items()})


@app.command("evaluate-risk")
def evaluate_risk(
    out: Path = typer.Option(Path("data/risk_eval_report.json")),
    track: bool = typer.Option(False, help=TRACK_HELP),
) -> None:
    """Weight-sensitivity of the risk ranking, and supplier link-prediction baselines."""
    report = pipeline.evaluate_risk(load_settings())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["sensitivity"]
    typer.echo(f"Sensitivity ({s['baseline_at_risk']} at-risk companies; each weight x0.5 "
               f"and x1.5): min top-{s['top_k']} overlap {s['min_top_k_overlap']:.2f}, "
               f"min Kendall tau {s['min_kendall_tau']:.2f}")
    lp = report["link_prediction"]
    typer.echo(f"\nLink prediction, leave-one-out: {lp['queries']} queries over "
               f"{lp['links']} links, ~{lp['candidates_per_query']} candidates each")
    typer.echo(f"{'method':<14}{'MRR':>7}{'hit@1':>8}{'hit@3':>8}")
    for name, m in lp["methods"].items():
        typer.echo(f"{name:<14}{m['mrr']:>7.3f}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}")
    typer.echo(f"Full report -> {out}")
    if track:
        from dataclasses import asdict

        from supplygraph.risk.engine import RiskConfig

        _track("risk", report, {f"risk.{k}": v for k, v in asdict(RiskConfig()).items()})


def _claude(call):
    """Run a Claude-backed pipeline step; turn setup/billing errors into one clear line."""
    import anthropic

    try:
        return call()
    except anthropic.APIStatusError as exc:
        body = exc.body if isinstance(exc.body, dict) else {}
        detail = (body.get("error") or {}).get("message") or str(exc)
        typer.echo(f"Claude API error ({exc.status_code}): {detail}", err=True)
    except TypeError as exc:  # raised by the SDK when no credentials are configured
        if "authentication" not in str(exc):
            raise
        typer.echo("No Claude credentials: set ANTHROPIC_API_KEY in .env (API usage needs "
                   "prepaid API credit; a claude.ai subscription does not cover it).", err=True)
    raise typer.Exit(code=2)


@app.command()
def ask(
    question: str = typer.Argument(..., help="A question about the supplier-risk graph."),
    read_only: bool = typer.Option(False, "--read-only", help="Don't offer action tools."),
    effort: str = typer.Option("medium", help="Claude effort: low, medium, high, xhigh."),
    use_graph: bool = typer.Option(False, "--graph",
                                   help="Add the multi-hop Cypher tools (Neo4j must be up)."),
) -> None:
    """Ask the graph agent (Claude + read-only graph tools). Answers cite graph refs;
    any action it takes is only a proposal awaiting `supplygraph actions approve`."""
    settings = load_settings()
    result = _claude(lambda: pipeline.ask(settings, question, allow_actions=not read_only,
                                          effort=effort, use_graph=use_graph))
    if result.error:
        typer.echo(f"ERROR: {result.error}", err=True)
    typer.echo(result.answer or "(no answer)")
    typer.echo(f"\n--- session {result.session_id}: {len(result.tool_calls)} tool calls, "
               f"{len(result.citations)} citations")
    if result.ungrounded:
        typer.echo(f"WARNING: cited refs no tool returned (treat as unverified): "
                   f"{', '.join(result.ungrounded)}", err=True)
    for p in result.proposals:
        what = f"{p['action']} {p['lei']}"
        if p["sanction_id"]:
            what += f" / {p['sanction_id']}"
        state = "proposed" if p["created"] else f"already {p['status']}"
        typer.echo(f"ACTION {p['action_id']} {state}: {what}  "
                   f"(supplygraph actions approve {p['action_id']})")
    typer.echo(f"Transcript -> {settings.agent_sessions_dir / (result.session_id + '.json')}")
    if result.error:
        raise typer.Exit(code=1)


actions_app = typer.Typer(help="Review the agent's proposed actions (audited).",
                          no_args_is_help=True)
app.add_typer(actions_app, name="actions")


@actions_app.command("list")
def actions_list(
    status: str | None = typer.Option(None, help="proposed, approved or rejected."),
) -> None:
    """List actions with their status and audit trail."""
    rows = pipeline.list_actions(load_settings(), status)
    if not rows:
        typer.echo("No actions.")
    for a in rows:
        target = a.lei + (f" / {a.sanction_id}" if a.sanction_id else "")
        typer.echo(f"{a.action_id}  {a.status:<9} {a.action:<15} {target}")
        typer.echo(f"    reason:   {a.reason}")
        typer.echo(f"    evidence: {', '.join(a.evidence)}")
        typer.echo(f"    proposed: {a.proposed_at} by {a.proposed_by}"
                   + (f" (session {a.session_id})" if a.session_id else ""))
        if a.decided_by:
            typer.echo(f"    {a.status}: {a.decided_at} by {a.decided_by}"
                       + (f" — {a.decision_note}" if a.decision_note else ""))


def _decide(action_id: str, approve: bool, by: str | None, note: str | None) -> None:
    from supplygraph.agent.actions import ActionError

    try:
        a = pipeline.decide_action(load_settings(), action_id, approve,
                                   by or getpass.getuser(), note)
    except ActionError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{a.action_id} {a.status} by {a.decided_by}: {a.action} {a.lei}"
               + (f" / {a.sanction_id}" if a.sanction_id else ""))
    if approve:
        step = "resolve" if a.sanction_id else "build"
        typer.echo(f"Takes effect at the next `supplygraph {step}` "
                   f"(then score-risk and load-graph).")


@actions_app.command("approve")
def actions_approve(
    action_id: str,
    by: str | None = typer.Option(None, help="Analyst name (default: OS user)."),
    note: str | None = typer.Option(None),
) -> None:
    """Approve a proposed action (re-checks its precondition against current data)."""
    _decide(action_id, True, by, note)


@actions_app.command("reject")
def actions_reject(
    action_id: str,
    note: str = typer.Option(..., help="Why the proposal is rejected."),
    by: str | None = typer.Option(None, help="Analyst name (default: OS user)."),
) -> None:
    """Reject a proposed action."""
    _decide(action_id, False, by, note)


@app.command("build-agent-fixture")
def build_agent_fixture(eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR / "agent")) -> None:
    """Snapshot the warehouse into the agent eval fixture (then re-check the cases)."""
    counts = pipeline.build_agent_fixture(load_settings(), eval_dir)
    typer.echo(f"Fixture -> {eval_dir / 'fixture'}: {counts}")
    problems = pipeline.check_agent_eval_set(eval_dir)["problems"]
    for p in problems:
        typer.echo(f"STALE CASE: {p}", err=True)
    if problems:
        raise typer.Exit(code=1)


@app.command("evaluate-agent")
def evaluate_agent(
    eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR / "agent"),
    case: list[str] | None = typer.Option(None, help="Only these case ids (repeatable)."),
    effort: str = typer.Option("medium"),
    check_only: bool = typer.Option(False, help="Offline: only check the gold is answerable."),
    judge: bool = typer.Option(False, help="Also grade each answer with an LLM judge against "
                                           "the evidence the agent retrieved."),
    out: Path = typer.Option(Path("data/agent_eval_report.json")),
    track: bool = typer.Option(False, help=TRACK_HELP),
) -> None:
    """Run the agent eval suite on the frozen fixture (calls Claude once per case)."""
    if check_only:
        result = pipeline.check_agent_eval_set(eval_dir)
        for p in result["problems"]:
            typer.echo(f"PROBLEM: {p}", err=True)
        typer.echo(f"{result['cases']} cases, {len(result['problems'])} problems")
        if result["problems"]:
            raise typer.Exit(code=1)
        return

    def progress(g: dict) -> None:
        failed = [k for k, ok in g["checks"].items() if not ok]
        typer.echo(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['id']:<32}"
                   f"{len(g['tool_calls']):>3} tools  {', '.join(failed)}")

    def run() -> dict:
        llm_judge = None
        if judge:
            from supplygraph.agent.judge import ClaudeJudge

            llm_judge = ClaudeJudge(effort=effort)
        return pipeline.evaluate_agent(load_settings(), eval_dir, effort, case,
                                       on_case=progress, judge=llm_judge)

    report = _claude(run)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    s = report["summary"]
    typer.echo(f"\n{report['model']} (effort {report['effort']}): {s['cases']} cases, "
               f"pass rate {s['pass_rate']}")
    typer.echo(f"citation recall {s['citation_recall']}  grounding {s['citation_grounding']}  "
               f"action precision {s['action_precision']} recall {s['action_recall']}  "
               f"unrequested-action cases {s['unrequested_action_cases']}")
    for cat, m in s["by_category"].items():
        typer.echo(f"  {cat:<18}{m['cases']:>3} cases  pass {m['pass_rate']}")
    if "judge" in s:
        j = s["judge"]
        typer.echo(f"judge {j['model']}: supported {j['supported_rate']}  answers question "
                   f"{j['answers_question_rate']}  mean score {j['mean_score']}  failed "
                   f"{j['judge_failed']}  agreement with rules {j['agreement_with_rules']} "
                   f"(kappa {j['cohen_kappa']})")
        for cid in j["rules_pass_judge_fail"]:
            typer.echo(f"  rules pass, judge fails: {cid}")
    typer.echo(f"tokens: {s['tokens']}\nFull report (answers, failed checks) -> {out}")
    if track:
        from supplygraph import prompts

        _track("agent", report, {
            "model": report["model"], "effort": report["effort"], "cases": s["cases"],
            "judge_model": s["judge"]["model"] if "judge" in s else None,
            **{f"prompt.{k}": v for k, v in prompts.versions().items()
               if k.startswith("agent.")}})


@app.command()
def prompts(
    update: bool = typer.Option(False, help="Rewrite prompts.lock.json from the code."),
) -> None:
    """List LLM prompt versions and check them against prompts.lock.json."""
    from supplygraph import prompts as registry

    current = registry.registry()
    if update:
        registry.write_lock(current)
    problems = registry.lock_problems(current, registry.read_lock())
    for name, p in sorted(current.items()):
        typer.echo(f"{name:<20}{p.version:<6}{p.fingerprint}")
    for problem in problems:
        typer.echo(f"PROBLEM: {problem}", err=True)
    if problems:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address. Off localhost, decisions and "
                                               "ask need SUPPLYGRAPH_API_TOKEN."),
    port: int = typer.Option(8000),
    static: Path = typer.Option(Path("web/dist"), help="Built UI (cd web; npm run build)."),
) -> None:
    """Serve the explorer UI and its read API (graph explorer, risk dashboard, actions)."""
    import os

    import uvicorn

    from supplygraph.api.app import app_for_settings, write_mode

    if not (static / "index.html").exists():
        typer.echo(f"No built UI in {static}: serving the API only (docs at /docs). "
                   "Build it with: cd web; npm install; npm run build", err=True)
    token = os.getenv("SUPPLYGRAPH_API_TOKEN") or None
    writes = write_mode(host, token)
    if writes == "off":
        typer.echo("Bound beyond localhost without SUPPLYGRAPH_API_TOKEN: serving read-only "
                   "(analyst decisions and ask are disabled).", err=True)
    typer.echo(f"SupplyGraph explorer -> http://{host}:{port}  (writes: {writes})")
    uvicorn.run(app_for_settings(load_settings(), static, writes, token), host=host, port=port,
                log_level="warning")


@app.command("demo-data")
def demo_data(
    eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR / "agent", help="Holds fixture/."),
    force: bool = typer.Option(False, help="Replace an existing warehouse."),
) -> None:
    """Create a warehouse from the committed real-data snapshot: a demo with no network,
    API keys or pipeline run."""
    try:
        counts = pipeline.demo_data(load_settings(), eval_dir, force=force)
    except FileExistsError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Demo warehouse -> {load_settings().warehouse_path}: {counts}")


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


@app.command("build-graph-bench")
def build_graph_bench(eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR)) -> None:
    """Generate the multi-hop benchmark questions (gold computed from the warehouse)."""
    counts = pipeline.build_graph_bench(load_settings(), eval_dir)
    typer.echo(f"Questions -> {eval_dir / 'graph_bench' / 'questions.jsonl'}: {counts}")


@app.command("evaluate-graph-bench")
def evaluate_graph_bench(
    eval_dir: Path = typer.Option(DEFAULT_EVAL_DIR),
    k: int = typer.Option(10, min=1, help="Chunks retrieved by vector RAG."),
    radius: int = typer.Option(2, min=1, max=3, help="GraphRAG neighbourhood radius."),
    track: bool = typer.Option(False, "--track", help=TRACK_HELP),
) -> None:
    """Multi-hop retrieval benchmark: vector RAG vs GraphRAG vs Cypher tools (Neo4j up)."""
    try:
        report = pipeline.evaluate_graph_bench(load_settings(), eval_dir, k=k, radius=radius)
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    from supplygraph.graph_bench import METHODS

    def fmt(x: float | None) -> str:
        return "-" if x is None else f"{x:.3f}"

    for label, s in [("all", report["summary"]), ("multi-hop", report["multi_hop"]),
                     *report["by_family"].items()]:
        typer.echo(f"\n{label} ({s['questions']} questions, {s['negatives']} with no answer)")
        typer.echo(f"  {'method':<14}{'recall':>8}{'precision':>11}{'exact':>8}"
                   f"{'false-alarm':>13}{'tokens':>9}{'ms':>8}")
        for m in METHODS:
            if m in s:
                v = s[m]
                typer.echo(f"  {m:<14}{fmt(v['recall']):>8}{fmt(v['precision']):>11}"
                           f"{fmt(v['exact']):>8}{fmt(v['false_alarm']):>13}"
                           f"{v['tokens_median']:>9}{v['ms_median']:>8.1f}")
    if track:
        _track("graph_bench", report, report["params"])


@app.command("mcp")
def mcp(
    use_graph: bool = typer.Option(False, "--graph",
                                   help="Add the multi-hop Cypher tools (Neo4j must be up)."),
) -> None:
    """Serve the read-only graph tools as an MCP server over stdio (for Claude Code, Claude
    Desktop or any MCP client). Logs go to stderr; stdout carries the protocol."""
    pipeline.serve_mcp(load_settings(), use_graph=use_graph)


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
    if load_settings().sec_user_agent:
        ingest_filings(filers=DEFAULT_FILERS)
        extract_suppliers(method="rules")
    else:
        typer.echo("SEC_USER_AGENT not set: skipping filings and supplier extraction.", err=True)
    ingest_news(hours=24)
    build()
    resolve(llm=False)
    score_risk()
    check()
    if not skip_graph:
        load_graph(batch_size=1000)


def _worker_module():
    """supplygraph.worker, or a clear error if the `worker` extra isn't installed."""
    try:
        from supplygraph import worker
    except ModuleNotFoundError as exc:
        typer.echo(f"Celery is not installed ({exc}); pip install -e .[worker]", err=True)
        raise typer.Exit(code=1) from exc
    return worker


@app.command("worker")
def run_worker(
    queues: str = typer.Option("warehouse,vectors", help="Comma-separated: warehouse, vectors."),
    concurrency: int = typer.Option(1, help="Task slots (the warehouse queue requires 1)."),
    pool: str = typer.Option(None, help="Celery pool (default: solo on Windows, else prefork)."),
) -> None:
    """Run a Celery worker for background jobs (needs Redis, CELERY_BROKER_URL)."""
    worker = _worker_module()
    try:
        argv = worker.worker_argv([q.strip() for q in queues.split(",") if q.strip()],
                                  concurrency, pool)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    worker.app.worker_main(argv)


@app.command("enqueue")
def enqueue(
    job: str = typer.Argument(..., help="resolve | index-vectors"),
    llm: bool = typer.Option(False, help="resolve: let Claude adjudicate the review band."),
    wait: bool = typer.Option(False, help="Block until the job finishes and print its result."),
    timeout: float = typer.Option(1800, help="Seconds to wait with --wait."),
) -> None:
    """Queue a background job; prints the task id (see `supplygraph job <id>`)."""
    worker = _worker_module()
    kwargs = {"use_llm": llm} if job == "resolve" else {}
    try:
        res = worker.submit(job, **kwargs)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(res.id)
    if wait:
        res.get(timeout=timeout, propagate=False)  # wait; the outcome is read via status()
        typer.echo(json.dumps(worker.status(res.id), indent=2, default=str))
        if res.failed():
            raise typer.Exit(code=1)


@app.command("job")
def job_status(task_id: str = typer.Argument(...)) -> None:
    """State (PENDING / STARTED / RETRY / SUCCESS / FAILURE) and result of a background job."""
    typer.echo(json.dumps(_worker_module().status(task_id), indent=2, default=str))


if __name__ == "__main__":
    app()
