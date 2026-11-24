"""Candidate generation (blocking) and scoring.

Comparing every company with every sanctions name is O(n*m). Instead, an inverted index maps
each informative token to the records containing it; a company is only scored against records
that share at least one such token. Very common tokens ("bank", "trading") are excluded from
blocking because they would make every bank a candidate for every other bank.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from supplygraph.er.normalize import normalize
from supplygraph.er.vectors import VectorIndex, embed

COUNTRY_MATCH = "match"
COUNTRY_MISMATCH = "mismatch"
COUNTRY_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Party:
    id: str
    names: tuple[str, ...]
    countries: frozenset[str] = frozenset()  # ISO alpha-2, upper case


@dataclass(frozen=True)
class MatcherConfig:
    translit: bool = True
    drop_legal_forms: bool = True
    max_token_frequency: int = 300  # tokens in more records than this are not used for blocking
    country_bonus: float = 0.05
    country_penalty: float = 0.10
    # Nearest sanctions names (per company name) added as candidates when a vector index is
    # given (er/vectors.py); 0 = token blocking only.
    vector_k: int = 0


@dataclass(frozen=True)
class Candidate:
    left_id: str
    right_id: str
    score: float
    name_score: float
    country_relation: str
    left_name: str
    right_name: str
    name_tokens: int = 2  # tokens in the shorter of the two compared (normalized) names

    @property
    def generic(self) -> bool:
        """A one-word match ("Atlant", "Aspect") is too weak to auto-approve: registries hold
        hundreds of unrelated companies with the same single-word name."""
        return self.name_tokens < 2

    def evidence(self) -> dict:
        return {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "name_score": round(self.name_score, 4),
            "country_relation": self.country_relation,
            "single_token_name": self.generic,
        }


@dataclass
class _Normalized:
    party: Party
    names: list[str] = field(default_factory=list)  # normalized, non-empty, de-duplicated
    originals: list[str] = field(default_factory=list)


def _prepare(party: Party, cfg: MatcherConfig) -> _Normalized:
    prepared = _Normalized(party)
    seen: set[str] = set()
    for original in party.names:
        norm = normalize(original, drop_legal_forms=cfg.drop_legal_forms, translit=cfg.translit)
        if norm and norm not in seen:
            seen.add(norm)
            prepared.names.append(norm)
            prepared.originals.append(original)
    return prepared


def name_similarity(a: str, b: str) -> float:
    """Order-insensitive similarity in [0, 1].

    token_set_ratio tolerates extra words ("rosneft" vs "rosneft oil") but scores any subset
    as 100, so it is only used when both names have 2+ tokens — otherwise "capital" would
    perfectly match "rt capital".
    """
    sort_score = fuzz.token_sort_ratio(a, b)
    if min(a.count(" "), b.count(" ")) == 0:
        return sort_score / 100.0
    return (0.6 * sort_score + 0.4 * fuzz.token_set_ratio(a, b)) / 100.0


def country_relation(a: frozenset[str], b: frozenset[str]) -> str:
    if not a or not b:
        return COUNTRY_UNKNOWN
    return COUNTRY_MATCH if a & b else COUNTRY_MISMATCH


class Matcher:
    def __init__(self, right: Iterable[Party], cfg: MatcherConfig | None = None,
                 vector_index: VectorIndex | None = None) -> None:
        self.cfg = cfg or MatcherConfig()
        self._right = [_prepare(p, self.cfg) for p in right]
        self._right = [p for p in self._right if p.names]
        index: dict[str, set[int]] = defaultdict(set)
        for i, prepared in enumerate(self._right):
            for name in prepared.names:
                for tok in name.split():
                    index[tok].add(i)
        self._index = {
            tok: ids for tok, ids in index.items() if len(ids) <= self.cfg.max_token_frequency
        }
        self._vectors = vector_index if self.cfg.vector_k > 0 else None
        if self._vectors is not None:
            self._vectors.build([(i, embed(name)) for i, prepared in enumerate(self._right)
                                 for name in prepared.names])

    def _block(self, left: _Normalized) -> set[int]:
        hits: set[int] = set()
        for name in left.names:
            for tok in name.split():
                hits |= self._index.get(tok, set())
        if self._vectors is not None and left.names:
            hits |= self._vectors.search([embed(n) for n in left.names], self.cfg.vector_k)
        return hits

    def candidates(self, party: Party, min_score: float = 0.0) -> list[Candidate]:
        """All scored candidates for one party, best first."""
        left = _prepare(party, self.cfg)
        results = []
        for i in self._block(left):
            right = self._right[i]
            best = (0.0, 0, 0)
            for li, ln in enumerate(left.names):
                for ri, rn in enumerate(right.names):
                    sim = name_similarity(ln, rn)
                    if sim > best[0]:
                        best = (sim, li, ri)
            name_score, li, ri = best
            relation = country_relation(party.countries, right.party.countries)
            score = name_score
            if relation == COUNTRY_MATCH:
                score += self.cfg.country_bonus
            elif relation == COUNTRY_MISMATCH:
                score -= self.cfg.country_penalty
            score = max(0.0, min(1.0, score))
            if score >= min_score:
                results.append(Candidate(
                    party.id, right.party.id, score, name_score, relation,
                    left.originals[li], right.originals[ri],
                    name_tokens=min(len(left.names[li].split()), len(right.names[ri].split())),
                ))
        results.sort(key=lambda c: (-c.score, c.right_id))
        return results

    def best(self, party: Party, min_score: float = 0.0) -> Candidate | None:
        found = self.candidates(party, min_score)
        return found[0] if found else None
