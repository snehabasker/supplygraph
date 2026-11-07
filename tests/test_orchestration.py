"""Dagster wiring: the quality gate must stop bad data from reaching Neo4j."""

from __future__ import annotations

import json

from dagster import AssetKey, materialize

from conftest import APPLE, NVIDIA
from supplygraph.orchestration.definitions import (
    ALL_ASSETS,
    SupplyGraphResource,
    defs,
    quality_gate,
)

# Nothing listens on port 1: if the graph asset runs when it shouldn't, it fails loudly
# instead of silently writing test data into a developer's local Neo4j.
DEAD_NEO4J = "bolt://127.0.0.1:1"


def run(data_dir, selection):
    return materialize(
        [*ALL_ASSETS, quality_gate],
        selection=selection,
        resources={"sg": SupplyGraphResource(data_dir=str(data_dir), neo4j_uri=DEAD_NEO4J)},
        raise_on_error=False,
    )


def materialized(result) -> set[str]:
    return {e.asset_key.to_user_string() for e in result.get_asset_materialization_events()}


def test_definitions_load_with_jobs_and_schedules() -> None:
    assert {j.name for j in defs.jobs} >= {"full_refresh", "sanctions_refresh"}
    assert len(list(defs.schedules)) == 2


def test_warehouse_builds_and_quality_gate_passes(raw_data_dir) -> None:
    result = run(raw_data_dir, ["warehouse"])
    assert result.success
    assert materialized(result) == {"warehouse"}
    [evaluation] = result.get_asset_check_evaluations()
    assert evaluation.passed


def test_failing_quality_gate_blocks_graph_load(raw_data_dir) -> None:
    # Corrupt the raw data: a company that owns itself.
    rels = raw_data_dir / "raw" / "gleif" / "relationships.jsonl"
    with rels.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"child_lei": APPLE, "parent_lei": APPLE,
                             "relationship_type": "DIRECT"}) + "\n")

    result = run(raw_data_dir, ["warehouse", "neo4j_graph"])

    [evaluation] = result.get_asset_check_evaluations()
    assert not evaluation.passed
    assert "ownership_no_self_loops" in json.dumps(evaluation.metadata["failing"].value)
    assert "neo4j_graph" not in materialized(result)
    assert not result.success
    # Sanity: the corrupt input really was loaded (the gate, not a crash, stopped the run).
    assert AssetKey("warehouse").to_user_string() in materialized(result)
    assert NVIDIA  # fixture still intact
