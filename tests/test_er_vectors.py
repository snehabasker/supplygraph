"""Vector candidate generation for entity resolution (er/vectors.py).

Unit tests use the in-memory ExactIndex; the pgvector test is opt-in
(SUPPLYGRAPH_INTEGRATION=1 and PGVECTOR_DSN, e.g. `docker compose up -d --wait pgvector`)
and only touches its own `test_er_vectors_*` table.
"""

from __future__ import annotations

import math
import os

import pytest

from supplygraph.er.matcher import Matcher, MatcherConfig, Party, name_similarity
from supplygraph.er.normalize import normalize
from supplygraph.er.vectors import ExactIndex, embed, rows_fingerprint


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_embedding_is_deterministic_and_unit_length() -> None:
    v = embed("gazprom neft")
    assert v == embed("gazprom neft")
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)
    assert embed("") == [0.0] * len(v)  # nothing to embed: zero vector, not a crash


def test_spacing_variants_embed_close_and_unrelated_names_far() -> None:
    assert cos(embed("gazpromneft"), embed("gazprom neft")) > 0.99
    assert cos(embed("rosatom"), embed("ros atom")) > 0.99
    assert cos(embed("gazpromneft"), embed("apple")) < 0.3


SANCTIONS = [
    Party("S-GPN", ("Gazprom Neft PJSC",), frozenset({"RU"})),
    Party("S-OTHER", ("Kama Trading House LLC",), frozenset({"RU"})),
]
COMPANY = Party("L1", ("Gazpromneft PAO",), frozenset({"RU"}))  # no whole token in common


def test_token_blocking_misses_a_spacing_variant_that_vector_blocking_finds() -> None:
    assert Matcher(SANCTIONS).candidates(COMPANY) == []
    matcher = Matcher(SANCTIONS, MatcherConfig(vector_k=1), ExactIndex())
    found = matcher.candidates(COMPANY)
    assert [c.right_id for c in found] == ["S-GPN"]
    # Vectors only widen the net: the candidate is scored by the usual name similarity.
    assert found[0].name_score == name_similarity(normalize("Gazpromneft PAO"),
                                                  normalize("Gazprom Neft PJSC"))


def test_vector_index_is_ignored_when_vector_k_is_zero() -> None:
    class Exploding(ExactIndex):
        def build(self, rows):  # pragma: no cover - must not be called
            raise AssertionError("index used although vector_k = 0")

    assert Matcher(SANCTIONS, MatcherConfig(), Exploding()).candidates(COMPANY) == []


def test_company_with_no_usable_name_gets_no_vector_candidates() -> None:
    """CJK-only GLEIF names normalize to nothing: vectors cannot help there (see docs)."""
    matcher = Matcher(SANCTIONS, MatcherConfig(vector_k=5), ExactIndex())
    assert matcher.candidates(Party("L2", ("中国海洋石油集团有限公司",))) == []


def test_fingerprint_changes_with_content() -> None:
    rows = [(0, embed("a b")), (1, embed("c d"))]
    assert rows_fingerprint(rows) == rows_fingerprint(list(rows))
    assert rows_fingerprint(rows) != rows_fingerprint(rows[:1])


@pytest.mark.skipif(os.getenv("SUPPLYGRAPH_INTEGRATION") != "1" or not os.getenv("PGVECTOR_DSN"),
                    reason="set SUPPLYGRAPH_INTEGRATION=1 and PGVECTOR_DSN with pgvector running")
def test_pgvector_index_matches_exact_search_and_skips_identical_rebuilds() -> None:
    from supplygraph.er.vectors import PgVectorIndex

    names = ["gazprom neft", "kama trading house", "rosatom", "severgaz bank", "atlant",
             "aspect capital", "vk company", "rosneft oil"]
    rows = [(i, embed(n)) for i, n in enumerate(names)]
    queries = [embed("gazpromneft"), embed("ros atom")]
    exact = ExactIndex()
    exact.build(rows)
    with PgVectorIndex(os.environ["PGVECTOR_DSN"], table="test_er_vectors_idx") as pg:
        try:
            pg.build(rows)
            assert pg.rebuilt
            assert pg.search(queries, 2) == exact.search(queries, 2)
            pg.build(rows)
            assert not pg.rebuilt  # same content: idempotent, no rebuild
            pg.build(rows[:4])
            assert pg.rebuilt and pg.search(queries, 10) <= {0, 1, 2, 3}
        finally:
            cur = pg._conn.cursor()
            cur.execute("DROP TABLE IF EXISTS test_er_vectors_idx")
            cur.execute("DELETE FROM er_vector_meta WHERE table_name = 'test_er_vectors_idx'")
