"""FastAPI app behind the explorer UI.

Read endpoints open a read-only DuckDB connection per request, so the UI never blocks or
changes the pipeline. The only writes are analyst decisions on proposed actions, which append
to the same audited event log as `supplygraph actions approve|reject`.

Writes (analyst decisions, and `/api/ask`, which spends API credit) have three modes, chosen
by `write_mode()` (docs/adr/0010):

* `open`  - no auth; only for a server bound to localhost (the `supplygraph serve` default);
* `token` - every write needs `Authorization: Bearer <SUPPLYGRAPH_API_TOKEN>`;
* `off`   - read-only; what a non-local server gets when no token is configured.

Reads are unauthenticated in every mode: put the server behind your network's access control
(ingress auth, VPN) if the data must not be public.

No `from __future__ import annotations` here: FastAPI resolves the dependency annotations
(`Con`, defined inside `create_app`) at runtime.
"""

import hmac
import logging
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import duckdb
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from supplygraph import __version__
from supplygraph.agent.actions import STATUSES, ActionError, ActionStore
from supplygraph.agent.tools import RELATIONS, GraphTools, ToolError
from supplygraph.api import views
from supplygraph.api.metrics import Metrics
from supplygraph.warehouse import table_exists

log = logging.getLogger(__name__)

Connect = Callable[[], duckdb.DuckDBPyConnection]
WriteMode = Literal["open", "token", "off"]
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def write_mode(host: str, token: str | None) -> WriteMode:
    """A token always protects writes; without one, writes are open only on localhost."""
    if token:
        return "token"
    return "open" if host in LOOPBACK else "off"


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
    writes: WriteMode = "open",
    token: str | None = None,
) -> FastAPI:
    if writes == "token" and not token:
        raise ValueError("write mode 'token' needs a token")
    app = FastAPI(title="SupplyGraph", version=__version__,
                  description="Supplier-risk knowledge graph: read API + analyst decisions.")
    store = ActionStore(actions_log)
    metrics = Metrics(connect, store)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path})
            response = JSONResponse({"detail": "internal error"}, status_code=500)
        route = request.scope.get("route")
        template = getattr(route, "path", "unmatched")
        if template.startswith("/api") or template in ("/healthz", "/readyz"):
            metrics.observe(request.method, template, response.status_code, started)
        return response

    def require_write(authorization: Annotated[str | None, Header()] = None) -> None:
        if writes == "open":
            return
        if writes == "off":
            raise HTTPException(403, "This server is read-only: start it with "
                                     "SUPPLYGRAPH_API_TOKEN set to enable decisions and ask.")
        scheme, _, given = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.encode(),
                                                                 (token or "").encode()):
            raise HTTPException(401, "Missing or wrong API token (Authorization: Bearer ...).",
                                headers={"WWW-Authenticate": "Bearer"})

    Writer = Annotated[None, Depends(require_write)]

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        """Liveness: the process serves requests."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> Response:
        """Readiness: the warehouse opens and has companies (otherwise there is nothing to
        serve; a pipeline run or `demo-data` fixes it)."""
        try:
            con = connect()
            try:
                ok = table_exists(con, "companies") and (
                    con.execute("SELECT count(*) FROM companies").fetchone()[0] > 0)
            finally:
                con.close()
        except Exception as exc:
            return JSONResponse({"status": "unavailable", "reason": str(exc)[:200]}, 503)
        if not ok:
            return JSONResponse({"status": "unavailable", "reason": "warehouse is empty"}, 503)
        return JSONResponse({"status": "ready"})

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/config")
    def config() -> dict:
        """What the UI needs to know about this server."""
        return {"version": __version__, "writes": writes}

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
    def decide(con: Con, action_id: str, decision: Decision, _: Writer) -> dict:
        try:
            a = store.decide(con, action_id, decision.approve, decision.by, decision.note)
        except ActionError as exc:
            code = 404 if str(exc).startswith("no action") else 409
            raise HTTPException(code, str(exc)) from exc
        return a.as_dict()

    @app.post("/api/ask")
    def ask(con: Con, body: Question, _: Writer) -> dict:
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


def app_for_settings(settings, static_dir: Path | None = None, writes: WriteMode = "open",
                     token: str | None = None) -> FastAPI:
    """The server starts even before the warehouse exists (e.g. a fresh deployment waiting for
    its first pipeline run): /readyz reports 503 until it does."""
    path = settings.warehouse_path
    if not path.exists():
        log.warning("warehouse %s not found yet; not ready until the pipeline creates it", path)
    return create_app(lambda: duckdb.connect(str(path), read_only=True), settings.actions_log,
                      static_dir=static_dir, sessions_dir=settings.agent_sessions_dir,
                      writes=writes, token=token)
