"""FastAPI app behind the explorer UI.

Read endpoints open a read-only DuckDB connection per request, so the UI never blocks or
changes the pipeline. The only writes are analyst decisions on proposed actions, which append
to the same audited event log as `supplygraph actions approve|reject`.

There is no authentication: the server is meant for one analyst on localhost
(`supplygraph serve` binds 127.0.0.1). Don't expose it on a network as-is.

No `from __future__ import annotations` here: FastAPI resolves the dependency annotations
(`Con`, defined inside `create_app`) at runtime.
"""

from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from supplygraph.agent.actions import STATUSES, ActionError, ActionStore
from supplygraph.agent.tools import RELATIONS, GraphTools, ToolError
from supplygraph.api import views

Connect = Callable[[], duckdb.DuckDBPyConnection]


class Decision(BaseModel):
    approve: bool
    by: str = Field(min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=2000)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    allow_actions: bool = True


def create_app(
    connect: Connect,
    actions_log: Path,
    *,
    now: datetime | None = None,
    static_dir: Path | None = None,
    sessions_dir: Path | None = None,
    agent_client: object | None = None,
) -> FastAPI:
    app = FastAPI(title="SupplyGraph", version="0.7",
                  description="Supplier-risk knowledge graph: read API + analyst decisions.")
    store = ActionStore(actions_log)

    def db() -> Iterator[duckdb.DuckDBPyConnection]:
        con = connect()
        try:
            yield con
        finally:
            con.close()

    Con = Annotated[duckdb.DuckDBPyConnection, Depends(db)]

    def tool(con: duckdb.DuckDBPyConnection, name: str, **args) -> dict:
        try:
            return GraphTools(con, now=now, store=store).call(name, args)
        except ToolError as exc:
            raise HTTPException(404 if "no company" in str(exc) or "no sanctions" in str(exc)
                                else 400, str(exc)) from exc

    @app.get("/api/summary")
    def summary(con: Con) -> dict:
        return views.summary(con, store)

    @app.get("/api/search")
    def search(con: Con, q: Annotated[str, Query(min_length=1)],
               limit: Annotated[int, Query(ge=1, le=20)] = 8) -> dict:
        return tool(con, "search_companies", query=q, limit=limit)

    @app.get("/api/risks")
    def risks(con: Con, limit: Annotated[int, Query(ge=1, le=50)] = 20,
              scope: Literal["total", "own", "inherited"] = "total") -> dict:
        return tool(con, "top_risks", limit=limit, scope=scope)

    @app.get("/api/review-queue")
    def review_queue(con: Con) -> dict:
        return tool(con, "review_queue")

    @app.get("/api/companies/{lei}")
    def company(con: Con, lei: str) -> dict:
        return tool(con, "get_company", lei=lei)

    @app.get("/api/companies/{lei}/relationships/{relation}")
    def relationships(con: Con, lei: str, relation: str,
                      limit: Annotated[int, Query(ge=1, le=100)] = 50) -> dict:
        if relation not in RELATIONS:
            raise HTTPException(400, f"relation must be one of {', '.join(RELATIONS)}")
        return tool(con, "get_relationships", lei=lei, relation=relation, limit=limit)

    @app.get("/api/companies/{lei}/news")
    def news(con: Con, lei: str, limit: Annotated[int, Query(ge=1, le=50)] = 20) -> dict:
        return tool(con, "get_news", lei=lei, limit=limit)

    @app.get("/api/companies/{lei}/graph")
    def graph(con: Con, lei: str, depth: Annotated[int, Query(ge=1, le=3)] = 1,
              max_nodes: Annotated[int, Query(ge=2, le=views.MAX_NODES)] = views.MAX_NODES
              ) -> dict:
        try:
            return views.neighborhood(con, lei, depth, max_nodes)
        except ToolError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/companies/{lei}/scenario")
    def scenario(con: Con, lei: str,
                 severity: Annotated[float, Query(ge=0, le=1)] = 1.0) -> dict:
        try:
            return views.scenario_graph(con, lei, severity, now=now)
        except ToolError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/sanctions/{sanction_id}")
    def sanction(con: Con, sanction_id: str) -> dict:
        return tool(con, "get_sanction", sanction_id=sanction_id)

    @app.get("/api/actions")
    def actions(status: str | None = None) -> dict:
        if status is not None and status not in STATUSES:
            raise HTTPException(400, f"status must be one of {', '.join(STATUSES)}")
        rows = [a.as_dict() for a in store.actions() if status is None or a.status == status]
        return {"actions": sorted(rows, key=lambda a: a["proposed_at"], reverse=True)}

    @app.post("/api/actions/{action_id}/decision")
    def decide(con: Con, action_id: str, decision: Decision) -> dict:
        try:
            a = store.decide(con, action_id, decision.approve, decision.by, decision.note)
        except ActionError as exc:
            code = 404 if str(exc).startswith("no action") else 409
            raise HTTPException(code, str(exc)) from exc
        return a.as_dict()

    @app.post("/api/ask")
    def ask(con: Con, body: Question) -> dict:
        import anthropic

        from supplygraph.agent.agent import Agent

        try:
            agent = Agent(con, store=store if body.allow_actions else None,
                          client=agent_client, now=now, sessions_dir=sessions_dir)
            return agent.ask(body.question).as_dict()
        except anthropic.APIStatusError as exc:
            detail = ((exc.body or {}).get("error") or {}).get("message") \
                if isinstance(exc.body, dict) else None
            raise HTTPException(502, f"Claude API error: {detail or exc}") from exc
        except TypeError as exc:
            if "authentication" not in str(exc):
                raise
            raise HTTPException(503, "No Claude credentials configured (ANTHROPIC_API_KEY); "
                                     "API usage needs prepaid API credit.") from exc

    if static_dir is not None and (static_dir / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, "not found")
            file = (static_dir / path).resolve()
            if path and file.is_file() and static_dir.resolve() in file.parents:
                return FileResponse(file)
            return FileResponse(static_dir / "index.html")

    return app


def app_for_settings(settings, static_dir: Path | None = None) -> FastAPI:
    path = settings.warehouse_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the pipeline first")
    return create_app(lambda: duckdb.connect(str(path), read_only=True), settings.actions_log,
                      static_dir=static_dir, sessions_dir=settings.agent_sessions_dir)
