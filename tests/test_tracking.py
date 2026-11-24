"""MLflow experiment tracking and the prompt registry (no network: a local SQLite store)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from supplygraph import cli, prompts, tracking


def test_flatten_metrics_keeps_numeric_leaves_and_sanitizes_keys() -> None:
    report = {
        "summary": {"pass_rate": 0.75, "cases": 4, "ok": True, "model": "x", "missing": None},
        "methods": {"popularity": {"hit@1": 0.5, "mrr": math.nan}},
        "cases": [{"id": "a", "score": 1.0}],  # per-case detail goes in the artifact
    }
    assert tracking.flatten_metrics(report) == {
        "summary.pass_rate": 0.75, "summary.cases": 4.0, "summary.ok": 1.0,
        "methods.popularity.hit_1": 0.5,
    }


def test_log_eval_records_params_metrics_series_and_report(tmp_path) -> None:
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    report = {"test": {"precision": 0.97, "recall": 0.76}, "misses": [{"lei": "X"}]}
    run_id = tracking.log_eval(
        "entity_resolution", report, tracking_uri=uri, artifact_root=tmp_path / "artifacts",
        params={"threshold.auto_match": 0.97, "blocking": "token", "model": None},
        series={"dev_sweep.recall": [(60, 0.9), (61, 0.88)]})

    client = mlflow.MlflowClient(uri)
    run = client.get_run(run_id)
    assert run.data.metrics["test.precision"] == 0.97
    assert run.data.params == {"threshold.auto_match": "0.97", "blocking": "token"}  # None dropped
    assert run.data.tags["eval"] == "entity_resolution"
    assert [(m.step, m.value) for m in client.get_metric_history(run_id, "dev_sweep.recall")] \
        == [(60, 0.9), (61, 0.88)]
    local = client.download_artifacts(run_id, "entity_resolution_report.json", str(tmp_path))
    assert json.loads(Path(local).read_text(encoding="utf-8"))["misses"] == [{"lei": "X"}]
    assert client.get_experiment(run.info.experiment_id).name == "supplygraph/entity_resolution"
    # A second run lands in the same experiment instead of failing to re-create it.
    second = tracking.log_eval("entity_resolution", report, tracking_uri=uri,
                               artifact_root=tmp_path / "artifacts")
    assert client.get_run(second).info.experiment_id == run.info.experiment_id


def test_cli_track_logs_the_news_cluster_eval(tmp_path, monkeypatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    result = CliRunner().invoke(cli.app, ["evaluate-news-clusters", "--track",
                                          "--out", str(tmp_path / "report.json")])
    assert result.exit_code == 0, result.output
    runs = mlflow.MlflowClient(uri).search_runs(
        [mlflow.MlflowClient(uri).get_experiment_by_name("supplygraph/news_clusters")
         .experiment_id])
    assert len(runs) == 1 and any(k.startswith("systems.") for k in runs[0].data.metrics)


# --- prompt registry -------------------------------------------------------------------------

def test_committed_prompt_lock_matches_the_code() -> None:
    """Fails when a prompt, tool description or output schema changes without a version bump
    (or the lock was not refreshed): run `supplygraph prompts --update` after bumping."""
    assert prompts.lock_problems(prompts.registry(), prompts.read_lock()) == []


def test_registry_covers_every_llm_prompt() -> None:
    assert set(prompts.registry()) == {"agent.system", "agent.judge", "er.adjudicate",
                                       "extract.suppliers"}


def test_fingerprint_covers_tool_specs_not_just_the_system_text() -> None:
    assert prompts.fingerprint("sys", [{"name": "a"}]) != prompts.fingerprint("sys",
                                                                              [{"name": "b"}])


def _lock(version: str, fp: str) -> dict:
    return {"p": {"version": version, "fingerprint": fp}}


@pytest.mark.parametrize(("current", "lock", "expect"), [
    (prompts.Prompt("p", "v1", "aaa"), _lock("v1", "aaa"), None),
    (prompts.Prompt("p", "v1", "bbb"), _lock("v1", "aaa"), "bump its PROMPT_VERSION"),
    (prompts.Prompt("p", "v2", "bbb"), _lock("v1", "aaa"), "lock is stale"),
    (prompts.Prompt("p", "v1", "aaa"), {}, "not in the lock file"),
])
def test_lock_problems(current, lock, expect) -> None:
    problems = prompts.lock_problems({"p": current}, lock)
    assert (problems == []) if expect is None else (expect in problems[0])


def test_removed_prompt_is_reported() -> None:
    assert "no longer exists" in prompts.lock_problems({}, _lock("v1", "a"))[0]


def test_prompts_cli_update_writes_a_lock_that_then_checks_clean(tmp_path, monkeypatch) -> None:
    lock = tmp_path / "prompts.lock.json"
    monkeypatch.setattr(prompts, "LOCK_PATH", lock)
    monkeypatch.setattr(prompts.read_lock, "__defaults__", (lock,))
    monkeypatch.setattr(prompts.write_lock, "__defaults__", (lock,))
    runner = CliRunner()
    assert runner.invoke(cli.app, ["prompts"]).exit_code == 1  # no lock yet
    assert runner.invoke(cli.app, ["prompts", "--update"]).exit_code == 0
    assert set(json.loads(lock.read_text(encoding="utf-8"))) == set(prompts.registry())
    assert runner.invoke(cli.app, ["prompts"]).exit_code == 0
