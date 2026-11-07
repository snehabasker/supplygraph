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

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def gleif_dir(self) -> Path:
        return self.raw_dir / "gleif"

    @property
    def sanctions_csv(self) -> Path:
        return self.raw_dir / "opensanctions" / "targets.simple.csv"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=Path(os.getenv("SUPPLYGRAPH_DATA_DIR", "data")),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "change-me-please"),
        opensanctions_url=os.getenv("OPENSANCTIONS_URL", DEFAULT_OPENSANCTIONS_URL),
    )
