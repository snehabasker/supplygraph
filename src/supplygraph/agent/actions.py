"""Audited analyst actions (docs/ontology.md, "Actions").

The agent never changes state directly: it *proposes* an action, and a human approves or
rejects it. Every step is an event appended to a JSONL log, which is the source of truth —
it lives outside the warehouse, so `build` (CREATE OR REPLACE) can never erase a human
decision. `build` folds the log into the `analyst_actions` table; `build_watchlist` and
`resolve` apply the approved actions.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from supplygraph.warehouse import table_exists

FLAG_SUPPLIER = "flag_supplier"
REQUEST_REVIEW = "request_review"
CONFIRM_MATCH = "confirm_match"
DISMISS_MATCH = "dismiss_match"
ACTIONS = (FLAG_SUPPLIER, REQUEST_REVIEW, CONFIRM_MATCH, DISMISS_MATCH)
MATCH_ACTIONS = frozenset({CONFIRM_MATCH, DISMISS_MATCH})

PROPOSED = "proposed"
APPROVED = "approved"
REJECTED = "rejected"
STATUSES = (PROPOSED, APPROVED, REJECTED)

# A citation / evidence reference to an ontology object: lei:<LEI>, sanction:<id>,
# filing:<accession>, news:<GDELT id>.
REF_KINDS = ("lei", "sanction", "filing", "news")
REF_PATTERN = re.compile(r"\b(lei|sanction|filing|news):([A-Za-z0-9][A-Za-z0-9._\-]*)")

MAX_REASON_LENGTH = 2000


class ActionError(ValueError):
    """An action that violates its precondition; the message is safe to show the agent."""


@dataclass
class Action:
    action_id: str
    action: str
    lei: str
    sanction_id: str | None
    reason: str
    evidence: list[str]
    proposed_by: str
    proposed_at: str
    session_id: str | None = None
    question: str | None = None
    status: str = PROPOSED
    decided_by: str | None = None
    decided_at: str | None = None
    decision_note: str | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str | None]:
        return self.action, self.lei, self.sanction_id

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("history")
        return d


def is_ref(value: str) -> bool:
    m = REF_PATTERN.fullmatch(value.strip())
    return m is not None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def validate(con: duckdb.DuckDBPyConnection, action: str, lei: str, sanction_id: str | None,
             reason: str) -> None:
    """Preconditions from the ontology. Raises ActionError with a message for the caller."""
    if action not in ACTIONS:
        raise ActionError(f"unknown action {action!r}; use one of {', '.join(ACTIONS)}")
    if not reason or not reason.strip():
        raise ActionError("a reason is required")
    if len(reason) > MAX_REASON_LENGTH:
        raise ActionError(f"reason is longer than {MAX_REASON_LENGTH} characters")
    if not con.execute("SELECT 1 FROM companies WHERE lei = ?", [lei]).fetchone():
        raise ActionError(f"no company with LEI {lei!r}")
    if action in MATCH_ACTIONS:
        if not sanction_id:
            raise ActionError(f"{action} needs a sanction_id")
        found = table_exists(con, "er_candidates") and con.execute(
            "SELECT 1 FROM er_candidates WHERE lei = ? AND sanction_id = ?", [lei, sanction_id]
        ).fetchone()
        if not found:
            raise ActionError(
                f"{lei} / {sanction_id} is not an entity-resolution candidate; only candidates "
                "can be confirmed or dismissed (identifier matches are facts in the source data)")
    elif sanction_id:
        raise ActionError(f"{action} takes no sanction_id")


class ActionStore:
    """Append-only event log of proposals and decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def actions(self) -> list[Action]:
        """Current state of every action, folded from the event log in order."""
        state: dict[str, Action] = {}
        for e in self.events():
            if e["event"] == PROPOSED:
                state[e["action_id"]] = Action(
                    action_id=e["action_id"], action=e["action"], lei=e["lei"],
                    sanction_id=e.get("sanction_id"), reason=e["reason"],
                    evidence=list(e.get("evidence") or []), proposed_by=e["actor"],
                    proposed_at=e["at"], session_id=e.get("session_id"),
                    question=e.get("question"))
            elif e["action_id"] in state:
                a = state[e["action_id"]]
                a.status, a.decided_by, a.decided_at = e["event"], e["actor"], e["at"]
                a.decision_note = e.get("note")
            else:
                continue  # a decision on an unknown action: ignored, never invents one
            state[e["action_id"]].history.append(e)
        return list(state.values())

    def get(self, action_id: str) -> Action:
        for a in self.actions():
            if a.action_id == action_id:
                return a
        raise ActionError(f"no action {action_id!r}")

    def _append(self, event: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def propose(self, con: duckdb.DuckDBPyConnection, *, action: str, lei: str,
                sanction_id: str | None, reason: str, evidence: list[str], proposed_by: str,
                session_id: str | None = None, question: str | None = None
                ) -> tuple[Action, bool]:
        """Record a proposal. Returns (action, created); an identical open or approved action
        is returned instead of being proposed twice."""
        lei = lei.strip().upper()
        sanction_id = (sanction_id or "").strip() or None
        validate(con, action, lei, sanction_id, reason)
        bad = [r for r in evidence if not is_ref(r)]
        if not evidence or bad:
            raise ActionError("evidence must be a non-empty list of refs like lei:<LEI>, "
                              f"sanction:<id>, filing:<accession>, news:<id>; got {bad or '[]'}")
        for existing in self.actions():
            if existing.key == (action, lei, sanction_id) and existing.status != REJECTED:
                return existing, False
        event = {"event_id": uuid.uuid4().hex, "action_id": f"A-{uuid.uuid4().hex[:10]}",
                 "event": PROPOSED, "at": _now(), "actor": proposed_by, "action": action,
                 "lei": lei, "sanction_id": sanction_id, "reason": reason.strip(),
                 "evidence": sorted(set(evidence)), "session_id": session_id,
                 "question": question}
        self._append(event)
        return self.get(event["action_id"]), True

    def decide(self, con: duckdb.DuckDBPyConnection, action_id: str, approve: bool,
               decided_by: str, note: str | None = None) -> Action:
        a = self.get(action_id)
        if a.status != PROPOSED:
            raise ActionError(f"{action_id} is already {a.status}")
        if not decided_by or not decided_by.strip():
            raise ActionError("the deciding analyst must be named")
        if approve:  # the data may have changed since the proposal
            validate(con, a.action, a.lei, a.sanction_id, a.reason)
        self._append({"event_id": uuid.uuid4().hex, "action_id": action_id,
                      "event": APPROVED if approve else REJECTED, "at": _now(),
                      "actor": decided_by.strip(), "note": note})
        return self.get(action_id)


ACTION_COLUMNS = {
    "action_id": "VARCHAR", "action": "VARCHAR", "lei": "VARCHAR", "sanction_id": "VARCHAR",
    "reason": "VARCHAR", "evidence": "VARCHAR", "status": "VARCHAR", "proposed_by": "VARCHAR",
    "proposed_at": "VARCHAR", "session_id": "VARCHAR", "question": "VARCHAR",
    "decided_by": "VARCHAR", "decided_at": "VARCHAR", "decision_note": "VARCHAR",
}


def build_actions_table(con: duckdb.DuckDBPyConnection, log_path: Path) -> int:
    """Fold the event log into `analyst_actions` (full refresh, idempotent)."""
    cols = ", ".join(f"{c} {t}" for c, t in ACTION_COLUMNS.items())
    con.execute(f"CREATE OR REPLACE TABLE analyst_actions ({cols})")
    rows = []
    for a in ActionStore(log_path).actions():
        d = a.as_dict()
        d["evidence"] = json.dumps(d["evidence"])
        rows.append([d[c] for c in ACTION_COLUMNS])
    if rows:
        con.executemany(f"INSERT INTO analyst_actions VALUES ({', '.join('?' * len(rows[0]))})",
                        rows)
    return len(rows)


def approved_match_decisions(con: duckdb.DuckDBPyConnection) -> dict[tuple[str, str], dict]:
    """(lei, sanction_id) -> the latest approved confirm/dismiss decision on that pair."""
    if not table_exists(con, "analyst_actions"):
        return {}
    out: dict[tuple[str, str], dict] = {}
    for action_id, action, lei, sid, reason, decided_by, decided_at in con.execute(
        "SELECT action_id, action, lei, sanction_id, reason, decided_by, decided_at "
        "FROM analyst_actions WHERE status = 'approved' AND action IN (?, ?) "
        "ORDER BY decided_at, action_id", [CONFIRM_MATCH, DISMISS_MATCH],
    ).fetchall():
        out[(lei, sid)] = {"action_id": action_id, "action": action, "reason": reason,
                           "decided_by": decided_by, "decided_at": decided_at}
    return out
