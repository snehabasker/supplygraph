from __future__ import annotations

import pytest

from conftest import APPLE, ROSNEFT, make_lei
from supplygraph import quality, warehouse
from supplygraph.graph import batched, fetch_dicts
from supplygraph.ingest import gleif


@pytest.fixture
def built(raw_data_dir):
    con = warehouse.connect(":memory:")
    counts = warehouse.build_all(
        con, raw_data_dir / "raw" / "gleif",
        raw_data_dir / "raw" / "opensanctions" / "targets.simple.csv",
    )
    yield con, counts, raw_data_dir
    con.close()


def test_make_lei_produces_valid_identifiers() -> None:
    assert gleif.is_valid_lei(make_lei("TESTSG000000000001"))


def test_build_counts(built) -> None:
    _, counts, _ = built
    assert counts == {
        "companies": 3, "ownership": 1, "sanctions": 3, "company_sanction_matches": 1,
    }


def test_lei_match_is_exact_token_not_substring(built) -> None:
    con, _, _ = built
    rows = fetch_dicts(con, "SELECT lei, sanction_id, match_method FROM company_sanction_matches")
    assert rows == [{"lei": ROSNEFT, "sanction_id": "NK-ros", "match_method": "LEI_IDENTIFIER"}]


def test_graph_rows_only_contain_neo4j_serializable_types(built) -> None:
    """Regression: DuckDB DECIMAL -> Python Decimal crashed the Neo4j driver."""
    con, _, _ = built
    allowed = (str, int, float, bool, type(None))
    for table in ("companies", "ownership", "company_sanction_matches"):
        for row in fetch_dicts(con, f"SELECT * FROM {table}"):
            bad = {k: type(v).__name__ for k, v in row.items() if not isinstance(v, allowed)}
            assert not bad, f"{table}: {bad}"


def test_build_is_idempotent(built) -> None:
    con, first, data_dir = built
    second = warehouse.build_all(
        con, data_dir / "raw" / "gleif", data_dir / "raw" / "opensanctions" / "targets.simple.csv"
    )
    assert first == second


def test_quality_checks_pass_on_clean_data(built) -> None:
    con, _, _ = built
    results = quality.run_checks(con)
    assert [r for r in results if r.status == "fail"] == []
    assert not quality.has_blocking_failures(results)


def test_quality_checks_catch_bad_data(built) -> None:
    con, _, _ = built
    con.execute(f"INSERT INTO companies (lei, legal_name) VALUES ('{APPLE}', 'dup')")
    con.execute(f"INSERT INTO ownership VALUES ('{APPLE}', '{APPLE}', 'DIRECT')")
    con.execute(f"INSERT INTO ownership VALUES ('{APPLE}', 'UNKNOWNLEI000000000', 'SIDEWAYS')")
    failed = {r.name for r in quality.run_checks(con) if r.status == "fail"}
    assert {
        "companies_lei_unique",
        "ownership_no_self_loops",
        "ownership_no_orphans",
        "ownership_type_valid",
    } <= failed
    assert quality.has_blocking_failures(quality.run_checks(con))


def test_checks_skip_missing_tables() -> None:
    con = warehouse.connect(":memory:")
    results = quality.run_checks(con)
    assert all(r.status == "skipped" for r in results)
    assert not quality.has_blocking_failures(results)


def test_missing_sanctions_file_gives_clear_error(tmp_path) -> None:
    con = warehouse.connect(":memory:")
    with pytest.raises(FileNotFoundError, match="ingest-sanctions"):
        warehouse.build_sanctions(con, tmp_path / "nope.csv")


def test_batched() -> None:
    rows = [{"i": i} for i in range(5)]
    assert [len(b) for b in batched(rows, 2)] == [2, 2, 1]
    with pytest.raises(ValueError):
        list(batched(rows, 0))
