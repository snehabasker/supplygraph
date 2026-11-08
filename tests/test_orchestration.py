"""Dagster wiring: the quality gates must stop bad data from reaching Neo4j."""

from __future__ import annotations

import json

from dagster import materialize

from conftest import APPLE
from supplygraph.orchestration.definitions import (
    ALL_ASSETS,
    ALL_CHECKS,
    SupplyGraphResource,
    defs,
)

# Nothing listens on port 1: if the graph asset runs when it shouldn't, it fails loudly
# instead of silently writing test data into a developer's local Neo4j.
DEAD_NEO4J = "bolt://127.0.0.1:1"


def run(data_dir, selection):
    return materialize(
        [*ALL_ASSETS, *ALL_CHECKS],
        selection=selection,
        resources={"sg": SupplyGraphResource(data_dir=str(data_dir), neo4j_uri=DEAD_NEO4J)},
        raise_on_error=False,
    )


def materialized(result) -> set[str]:
    return {e.asset_key.to_user_string() for e in result.get_asset_materialization_events()}


def test_definitions_load_with_jobs_and_schedules() -> None:
    assert {j.name for j in defs.jobs} >= {"full_refresh", "sanctions_refresh"}
    assert len(list(defs.schedules)) == 2


def test_warehouse_and_entity_resolution_pass_both_gates(raw_data_dir) -> None:
    result = run(raw_data_dir, ["warehouse", "entity_resolution"])
    assert result.success
    assert materialized(result) == {"warehouse", "entity_resolution"}
    evaluations = result.get_asset_check_evaluations()
    assert len(evaluations) == 2
    assert all(e.passed for e in evaluations)


def test_failing_quality_gate_blocks_everything_downstream(raw_data_dir) -> None:
    # Corrupt the raw data: a company that owns itself.
    rels = raw_data_dir / "raw" / "gleif" / "relationships.jsonl"
    with rels.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"child_lei": APPLE, "parent_lei": APPLE,
                             "relationship_type": "DIRECT"}) + "\n")

    result = run(raw_data_dir, ["warehouse", "entity_resolution", "neo4j_graph"])

    [evaluation] = [e for e in result.get_asset_check_evaluations()
                    if e.check_name == "quality_gate"]
    assert not evaluation.passed
    assert "ownership_no_self_loops" in json.dumps(evaluation.metadata["failing"].value)
    # The gate (not a crash) stopped the run: warehouse built, nothing downstream ran.
    assert materialized(result) == {"warehouse"}
    assert not result.success
