"""Prompt registry: every LLM prompt has a declared version and a content fingerprint.

A prompt is more than its system text: tool descriptions and the structured-output schema
steer the model just as much. The fingerprint covers all of it. `prompts.lock.json` records
(version, fingerprint) per prompt, and the unit tests fail when

* a prompt's content changed but its version was not bumped (silent prompt drift — cached
  answers and eval numbers would be attributed to the wrong prompt), or
* the version was bumped but the lock was not refreshed (`supplygraph prompts --update`).

Versions are logged as MLflow params with every eval run (`--track`) and are part of the
cache key for LLM extraction, so a result can always be traced to the prompt that made it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCK_PATH = Path(__file__).resolve().parents[2] / "prompts.lock.json"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    fingerprint: str


def fingerprint(*parts: Any) -> str:
    """Stable short hash of prompt text, tool specs and output schemas."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def registry() -> dict[str, Prompt]:
    """All prompts in the codebase. Imported lazily: the LLM modules need `pydantic`."""
    from supplygraph.agent import agent, cypher, judge, tools
    from supplygraph.er import adjudicate
    from supplygraph.extract import suppliers

    entries = [
        Prompt("agent.system", agent.PROMPT_VERSION,
               fingerprint(agent.SYSTEM_PROMPT, tools.TOOL_SPECS, agent.ACTION_SPECS,
                           cypher.CYPHER_TOOL_SPECS)),
        Prompt("agent.judge", judge.PROMPT_VERSION,
               fingerprint(judge.SYSTEM_PROMPT, judge.JudgeVerdict.model_json_schema())),
        Prompt("er.adjudicate", adjudicate.PROMPT_VERSION,
               fingerprint(adjudicate.SYSTEM_PROMPT, adjudicate.MatchVerdict.model_json_schema())),
        Prompt("extract.suppliers", suppliers.PROMPT_VERSION,
               fingerprint(suppliers.SYSTEM_PROMPT,
                           suppliers.PassageRelations.model_json_schema())),
    ]
    return {p.name: p for p in entries}


def versions() -> dict[str, str]:
    """{prompt name: version} — what eval runs log as params."""
    return {name: p.version for name, p in registry().items()}


def read_lock(path: Path = LOCK_PATH) -> dict[str, dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_lock(prompts: dict[str, Prompt], path: Path = LOCK_PATH) -> None:
    lock = {name: {"version": p.version, "fingerprint": p.fingerprint}
            for name, p in sorted(prompts.items())}
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def lock_problems(prompts: dict[str, Prompt], lock: dict[str, dict[str, str]]) -> list[str]:
    problems = []
    for name, p in sorted(prompts.items()):
        locked = lock.get(name)
        if locked is None:
            problems.append(f"{name}: not in the lock file; run `supplygraph prompts --update`")
        elif p.fingerprint != locked["fingerprint"] and p.version == locked["version"]:
            problems.append(f"{name}: content changed but version is still {p.version}; "
                            "bump its PROMPT_VERSION, then `supplygraph prompts --update`")
        elif p.fingerprint != locked["fingerprint"] or p.version != locked["version"]:
            problems.append(f"{name}: lock is stale ({locked['version']} -> {p.version}); "
                            "run `supplygraph prompts --update`")
    problems += [f"{name}: in the lock file but no longer exists" for name in sorted(lock)
                 if name not in prompts]
    return problems
