"""Experiment tracking: log every `evaluate-*` run to MLflow.

Each evaluation already writes a JSON report; that answers "what are the numbers now?" but not
"what were they three prompt versions ago?". With `--track`, the run is also logged to MLflow:

* **params**  — what produced the numbers: git commit, model, prompt versions, thresholds;
* **metrics** — every numeric leaf of the report, flattened (`test.auto_match_only.recall`);
* **series**  — optional stepped metrics (e.g. precision/recall over the threshold sweep);
* **artifact** — the full JSON report, so false alarms and misses travel with the run.

The tracking URI comes from `MLFLOW_TRACKING_URI`, defaulting to a local SQLite store under the
data dir (`mlflow ui --backend-store-uri sqlite:///data/mlflow.db` to browse). MLflow is an
optional dependency (`pip install -e .[tracking]`); nothing else imports it.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPERIMENT_PREFIX = "supplygraph"
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")
MAX_PARAM_CHARS = 6000  # MLflow's limit on a param value


def metric_key(path: str) -> str:
    """MLflow metric names allow only [A-Za-z0-9_-. /]; `hit@1` becomes `hit_1`."""
    return _INVALID_KEY_CHARS.sub("_", path)


def flatten_metrics(report: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Every finite numeric leaf of a nested report, keyed by its dotted path.

    Lists are skipped (they hold per-case detail, logged with the artifact instead); bools
    become 0/1 so pass/fail checks can be charted; None and NaN are dropped.
    """
    out: dict[str, float] = {}
    for key, value in report.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten_metrics(value, f"{path}."))
        elif isinstance(value, int | float) and math.isfinite(value):  # bool is an int
            out[metric_key(path)] = float(value)
    return out


def git_commit(cwd: Path | None = None) -> tuple[str | None, bool]:
    """(commit sha, working tree dirty?) — (None, False) outside a git checkout."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                    capture_output=True, text=True, check=True,
                                    timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, False
    return sha, dirty


def default_tracking_uri(data_dir: Path) -> str:
    return os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(data_dir / 'mlflow.db').as_posix()}"


def _param(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return text[:MAX_PARAM_CHARS]


def log_eval(
    name: str,
    report: Mapping[str, Any],
    *,
    params: Mapping[str, Any] | None = None,
    series: Mapping[str, list[tuple[int, float]]] | None = None,
    tracking_uri: str,
    artifact_root: Path | None = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Log one evaluation run to experiment `supplygraph/<name>`; returns the run id."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep CLI output to our report
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = f"{EXPERIMENT_PREFIX}/{name}"
    if artifact_root is not None and mlflow.get_experiment_by_name(experiment) is None:
        # Keep artifacts next to the store, not in ./mlruns of whatever directory we run from.
        mlflow.create_experiment(experiment,
                                 artifact_location=(artifact_root / name).resolve().as_uri())
    mlflow.set_experiment(experiment)
    sha, dirty = git_commit()
    run_tags = {"eval": name, "git_dirty": str(dirty).lower(), **(tags or {})}
    if sha:
        run_tags["mlflow.source.git.commit"] = sha
    with mlflow.start_run(tags=run_tags) as run:
        if params:
            mlflow.log_params({k: _param(v) for k, v in params.items() if v is not None})
        metrics = flatten_metrics(report)
        if metrics:
            mlflow.log_metrics(metrics)
        for key, points in (series or {}).items():
            for step, value in points:
                if math.isfinite(value):
                    mlflow.log_metric(metric_key(key), value, step=step)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}_report.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
            mlflow.log_artifact(str(path))
        return run.info.run_id
