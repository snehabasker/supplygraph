"""Vector candidate generation for entity resolution (pgvector).

Token blocking (matcher.py) only compares names that share a whole token, so it misses pairs
whose spelling differs *inside* the tokens: "Gazpromneft" / "Gazprom Neft", "Rosatom" /
"Ros Atom", transliteration variants that survive normalization. The eval measures this as
blocking recall < 1: those gold records never become candidates, whatever the threshold.

This module embeds every normalized sanctions name as a hashed character-trigram vector and
adds each company name's k nearest neighbours (cosine) to its candidate set. Candidates are
still scored by the same `name_similarity`, so thresholds keep their meaning: vectors only
widen the net, they never decide a match.

Why trigram vectors rather than a neural embedding model: names are lexical, not semantic
("Apple" the company is not near "Pear"), the vectors are deterministic and need no model or
API, and the gold set measures whether they help.

Two interchangeable indexes:
* `ExactIndex` — brute-force, in memory; for unit tests and small inputs;
* `PgVectorIndex` — PostgreSQL + pgvector with an HNSW index; rebuilt only when the indexed
  names change (fingerprint), so reruns are idempotent and cheap.
"""

from __future__ import annotations

import hashlib
import math
from array import array
from collections.abc import Sequence
from typing import Protocol

DIM = 256
DEFAULT_TABLE = "er_sanction_name_vectors"


def _bucket(gram: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)  # signed hashing reduces collision bias


def embed(name: str, dim: int = DIM) -> list[float]:
    """L2-normalized hashed character-trigram counts of an already normalized name.

    Spaces are dropped before n-gramming, so "gazprom neft" and "gazpromneft" embed alike;
    word boundaries still contribute through the padded start and end.
    """
    text = f"#{name.replace(' ', '')}#"
    vec = [0.0] * dim
    for i in range(len(text) - 2):
        idx, sign = _bucket(text[i:i + 3], dim)
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class VectorIndex(Protocol):
    def build(self, rows: Sequence[tuple[int, list[float]]]) -> None:
        """Index (record position, vector) rows, replacing any previous content."""

    def search(self, vectors: Sequence[list[float]], k: int) -> set[int]:
        """Record positions among the k nearest neighbours of any of `vectors`."""


def rows_fingerprint(rows: Sequence[tuple[int, list[float]]]) -> str:
    h = hashlib.sha256()
    for idx, vec in rows:
        h.update(idx.to_bytes(8, "big", signed=True))
        h.update(array("f", vec).tobytes())
    return h.hexdigest()[:32]


class ExactIndex:
    def __init__(self) -> None:
        self._rows: list[tuple[int, list[float]]] = []

    def build(self, rows: Sequence[tuple[int, list[float]]]) -> None:
        self._rows = list(rows)

    def search(self, vectors: Sequence[list[float]], k: int) -> set[int]:
        hits: set[int] = set()
        for q in vectors:
            scored = sorted(self._rows, key=lambda r: -sum(a * b for a, b in zip(q, r[1],
                                                                                  strict=True)))
            hits |= {idx for idx, _ in scored[:k]}
        return hits


def _literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


class PgVectorIndex:
    """pgvector table + HNSW (cosine) index. Requires `psycopg` and the `vector` extension."""

    def __init__(self, dsn: str, table: str = DEFAULT_TABLE, dim: int = DIM,
                 ef_search: int = 100) -> None:
        import psycopg

        if not table.isidentifier():
            raise ValueError(f"invalid table name {table!r}")
        self.table, self.dim, self.ef_search = table, dim, ef_search
        self._conn = psycopg.connect(dsn, autocommit=True)
        self.rebuilt = False

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PgVectorIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def build(self, rows: Sequence[tuple[int, list[float]]]) -> None:
        fp = rows_fingerprint(rows)
        cur = self._conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE TABLE IF NOT EXISTS er_vector_meta "
                    "(table_name text PRIMARY KEY, fingerprint text NOT NULL, rows int NOT NULL)")
        cur.execute("SELECT fingerprint FROM er_vector_meta WHERE table_name = %s", [self.table])
        found = cur.fetchone()
        if found and found[0] == fp:
            self.rebuilt = False  # same names, same vectors: keep the existing index
            return
        with self._conn.transaction():
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(f"CREATE TABLE {self.table} "
                        f"(id bigserial PRIMARY KEY, record_idx int NOT NULL, "
                        f"embedding vector({self.dim}) NOT NULL)")
            with cur.copy(f"COPY {self.table} (record_idx, embedding) FROM STDIN") as copy:
                for idx, vec in rows:
                    copy.write_row((idx, _literal(vec)))
            cur.execute(f"CREATE INDEX ON {self.table} "
                        f"USING hnsw (embedding vector_cosine_ops)")
            cur.execute("INSERT INTO er_vector_meta VALUES (%s, %s, %s) ON CONFLICT (table_name) "
                        "DO UPDATE SET fingerprint = EXCLUDED.fingerprint, rows = EXCLUDED.rows",
                        [self.table, fp, len(rows)])
        cur.execute(f"ANALYZE {self.table}")
        self.rebuilt = True

    def search(self, vectors: Sequence[list[float]], k: int) -> set[int]:
        hits: set[int] = set()
        cur = self._conn.cursor()
        # HNSW returns at most ef_search rows; keep it above k.
        cur.execute(f"SET hnsw.ef_search = {max(self.ef_search, 2 * k)}")
        for vec in vectors:
            cur.execute(f"SELECT record_idx FROM {self.table} "
                        f"ORDER BY embedding <=> %s::vector LIMIT %s", [_literal(vec), k])
            hits |= {row[0] for row in cur.fetchall()}
        return hits
