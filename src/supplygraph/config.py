"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_OPENSANCTIONS_URL = (
    "https://data.opensanctions.org/datasets/latest/sanctions/targets.simple.csv"
)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    opensanctions_url: str
    gleif_base_url: str = "https://api.gleif.org/api/v1"
    user_agent: str = "supplygraph/0.1"
    # SEC requires a User-Agent with a contact email; unset means SEC ingestion is unavailable.
    sec_user_agent: str | None = None
    # Where analyst actions are logged; defaults to <data_dir>/actions/events.jsonl. Set it when
    # the pipeline runs in a different data dir than the server (see deploy/), so both share
    # one audit log.
    actions_log_path: Path | None = None
    # PostgreSQL + pgvector, for vector candidate generation in entity resolution (optional).
    pgvector_dsn: str | None = None
    # Celery background jobs (supplygraph/worker.py): Redis as broker and result backend.
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str | None = None  # None = same as the broker
    # Agent / MCP graph tools (agent/cypher.py). A reader-role account where the edition has
    # RBAC (Enterprise / Aura); unset falls back to the loader account, still in READ sessions.
    neo4j_read_user: str | None = None
    neo4j_read_password: str | None = None
    neo4j_query_timeout: float = 5.0

    @property
    def neo4j_read_auth(self) -> tuple[str, str]:
        if self.neo4j_read_user:
            return self.neo4j_read_user, self.neo4j_read_password or ""
        return self.neo4j_user, self.neo4j_password

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def gleif_dir(self) -> Path:
        return self.raw_dir / "gleif"

    @property
    def supplier_gleif_dir(self) -> Path:
        """GLEIF records of suppliers found in filings (and their parents)."""
        return self.gleif_dir / "suppliers"

    @property
    def sec_dir(self) -> Path:
        return self.raw_dir / "sec"

    @property
    def gdelt_dir(self) -> Path:
        return self.raw_dir / "gdelt"

    @property
    def sanctions_csv(self) -> Path:
        return self.raw_dir / "opensanctions" / "targets.simple.csv"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    @property
    def actions_log(self) -> Path:
        """Append-only audit log of proposed / approved / rejected analyst actions."""
        return self.actions_log_path or self.data_dir / "actions" / "events.jsonl"

    @property
    def agent_sessions_dir(self) -> Path:
        return self.data_dir / "agent_sessions"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=Path(os.getenv("SUPPLYGRAPH_DATA_DIR", "data")),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "change-me-please"),
        opensanctions_url=os.getenv("OPENSANCTIONS_URL", DEFAULT_OPENSANCTIONS_URL),
        sec_user_agent=os.getenv("SEC_USER_AGENT") or None,
        actions_log_path=Path(p) if (p := os.getenv("SUPPLYGRAPH_ACTIONS_LOG")) else None,
        pgvector_dsn=os.getenv("PGVECTOR_DSN") or None,
        celery_broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        celery_result_backend=os.getenv("CELERY_RESULT_BACKEND") or None,
        neo4j_read_user=os.getenv("NEO4J_READ_USER") or None,
        neo4j_read_password=os.getenv("NEO4J_READ_PASSWORD") or None,
        neo4j_query_timeout=float(os.getenv("NEO4J_QUERY_TIMEOUT", "5")),
    )
