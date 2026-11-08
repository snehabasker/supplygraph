"""Turn an SEC filing (inline-XBRL HTML) into sentences, learn the short names it defines for
companies, and pick out the passages worth sending to an extractor.

Filings differ wildly in markup: 10-Ks usually put a paragraph in one <div>, while many 20-Fs
put every *line* in its own element. So we ignore the markup structure, flatten to one stream
of text and split sentences ourselves.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_DROP_BLOCKS = re.compile(r"(?is)<(script|style|ix:header)\b.*?</\1>")
_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")

# Bullets start a new sentence even without terminal punctuation.
_BULLET = re.compile(r"\s*[\u2022\u25cf\u25aa\u25e6\u2023\u2043]\s*")
# A sentence ends at . ! ? ; followed by whitespace and an uppercase letter, quote or bracket.
_BOUNDARY = re.compile(r"(?<=[.!?;])\s+(?=[\"\u201c(\[]?[A-Z])")
# ...unless the period belongs to an abbreviation that is common inside company names.
_ABBREVIATIONS = re.compile(
    r"(?:\b(?:Inc|Ltd|Co|Corp|Cos|No|Nos|Mr|Ms|Mrs|Dr|St|Jr|Sr|vs|U\.S|U\.K|N\.V|S\.A|"
    r"S\.p\.A|B\.V|L\.P|P\.L\.C|A\.G|S\.E|e\.g|i\.e|approx)\.)$"
)

_SUFFIX = (
    r"(?:Inc\.?|Incorporated|Corporation|Corp\.|Co\.,?\s?Ltd\.?|Company,?\s+Limited|Company|"
    r"Limited|Ltd\.?|N\.V\.|B\.V\.|GmbH|AG|plc|PLC|S\.A\.|SE|LLC|L\.P\.|Pte\.?\s+Ltd\.?|K\.K\.)"
)
# A capitalized name ending in a corporate suffix, e.g. "Hon Hai Precision Industry Co., Ltd."
# or "ASML Holding N.V.". The lookahead stops "Inc" matching inside "Income" and "SE" inside
# "SECURITIES" (both happened on real filings).
ORG_NAME = re.compile(
    r"\b[A-Z][\w&.\-']*(?:\s+(?:[A-Z][\w&.\-']*|of|and|&|de|der))*,?\s+" + _SUFFIX
    + r"(?![\w])"
)
# "The Company", "Our Company" etc. are the filer talking about itself.
_NOT_A_NAME = frozenset({"the", "our", "this", "such", "each", "any", "a", "an", "its", "parent"})

# Words that make a sentence relevant to who supplies the filer.
SUPPLY_CUE = re.compile(
    r"(?i)\b(suppl(?:y|ier|iers|ied|ies)|foundr(?:y|ies)|contract\s+manufactur\w*|"
    r"subcontract\w*|purchase[sd]?|procure[sd]?|sourced?|vendors?|outsourc\w*|"
    r"manufactured\s+(?:by|for\s+us)|assembly\s+partners?|wafers?|cells)\b"
)

# Definitions of a short name right after a full name:
#   "... Company Limited (“TSMC”)", "... Inc. (GF)", "... Co., Ltd., or Samsung, ..."
_ALIAS_AFTER = re.compile(
    # No \A: `.match(text, pos)` already anchors at pos, and \A would only match at index 0.
    r"(?:\s*\(\s*(?:the\s+)?[\"“]?(?P<paren>[A-Z][A-Za-z0-9&\-]{1,24})[\"”]?\s*\)"
    r"|,?\s+or\s+(?P<or>[A-Z][A-Za-z0-9&\-]{1,24})(?=[,.;)]))"
)


def html_to_text(raw_html: str) -> str:
    text = _DROP_BLOCKS.sub(" ", raw_html)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text).replace("\N{NO-BREAK SPACE}", " ")
    return _SPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for chunk in _BULLET.split(text):
        start = 0
        for m in _BOUNDARY.finditer(chunk):
            head = chunk[start:m.start()]
            if _ABBREVIATIONS.search(head):
                continue  # "Co. Ltd." / "Inc. (AMD)" are not sentence ends
            sentences.append(head.strip())
            start = m.end()
        sentences.append(chunk[start:].strip())
    return [s for s in sentences if s]


_SELF_REFERENCE = re.compile(r"\b(?:the|our|this|parent)\s+company$", re.I)
_SUFFIX_END = re.compile(_SUFFIX + r"(?![\w])")
_LIST_SEPARATOR = re.compile(r"\s*,?\s*(?:and\s+|&\s+)?")


def split_name_list(match: str) -> list[str]:
    """ORG_NAME is greedy, so "SK Hynix Inc. and Kioxia Corporation" is one match; cut it
    after every corporate suffix that is followed by more capitalized words."""
    parts, start = [], 0
    for m in _SUFFIX_END.finditer(match):
        rest = match[m.end():]
        sep = _LIST_SEPARATOR.match(rest)
        tail = rest[sep.end():] if sep else rest
        # Only cut if a complete name follows: "...Company Limited" is one name, not two.
        if tail[:1].isupper() and ORG_NAME.match(tail):
            parts.append(match[start:m.end()])
            start = m.end() + (sep.end() if sep else 0)
    parts.append(match[start:])
    return [p.strip().rstrip(",") for p in parts if p.strip()]


def org_names(sentence: str) -> list[str]:
    """Company names with a corporate suffix, minus "The Company"-style self references —
    including the ones glued to a section heading ("Accrued Liabilities The Company")."""
    names = []
    for m in ORG_NAME.finditer(sentence):
        names.extend(_clean_names(split_name_list(m.group(0))))
    return names


def _clean_names(candidates: list[str]) -> list[str]:
    names = []
    for name in candidates:
        name = name.strip().rstrip(",")
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", name)]
        if words and words[0] in _NOT_A_NAME and len(words) <= 3:
            continue
        if _SELF_REFERENCE.search(name):
            continue
        names.append(name)
    return names


def last_name_in_list(name: str) -> str:
    """ORG_NAME is greedy over capitalized words, so "Panasonic and Contemporary Amperex
    Technology Co. Limited" comes back as one match; the suffix belongs to the last one."""
    return name.rsplit(" and ", 1)[-1].strip()


def _trim_to_alias(name: str, alias: str) -> str:
    """Drop leading words while the alias still fits: "Basis of Presentation Overview
    Broadcom Inc." -> "Broadcom Inc."; "Contemporary Amperex Technology Co. Limited" stays."""
    words = last_name_in_list(name).split()
    while len(words) > 1 and _plausible_alias(alias, " ".join(words[1:])):
        words = words[1:]
    return " ".join(words)


def learn_aliases(text: str) -> dict[str, str]:
    """Short names the filing itself defines: {"TSMC": "Taiwan Semiconductor Manufacturing
    Company Limited", ...}. A definition is only trusted if the alias is an acronym of the
    name, or one of the name's own words ("Samsung" for "Samsung Electronics Co., Ltd.")."""
    from supplygraph.er.normalize import normalize

    found: dict[str, Counter] = {}
    for m in ORG_NAME.finditer(text):
        d = _ALIAS_AFTER.match(text, m.end())
        if not d:
            continue
        alias = d.group("paren") or d.group("or")
        if not alias or not normalize(alias):  # "PLC", "Inc" are legal forms, not names
            continue
        name = m.group(0).strip().rstrip(",")
        if _plausible_alias(alias, last_name_in_list(name)):
            found.setdefault(alias, Counter())[_trim_to_alias(name, alias)] += 1
    return {alias: names.most_common(1)[0][0] for alias, names in found.items()}


def _plausible_alias(alias: str, name: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", name)
    if alias.lower() in {w.lower() for w in words}:
        return True
    if not alias.isupper():
        return False
    initials = "".join(w[0] for w in words if w[0].isupper())
    it = iter(initials)  # alias letters appear in order among the initials (CATL, TSMC, GF)
    return all(ch in it for ch in alias if ch.isalpha())


def merge_aliases(per_filing: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """Aliases learned anywhere apply everywhere: Intel says "TSMC" but never defines it."""
    votes: dict[str, Counter] = {}
    for aliases in per_filing:
        for alias, name in aliases.items():
            votes.setdefault(alias, Counter())[name] += 1
    return {alias: names.most_common(1)[0][0] for alias, names in sorted(votes.items())}


@dataclass(frozen=True)
class Passage:
    id: str
    filer_cik: str
    text: str
    # The sentence indices covered, so overlapping windows can be merged and audited.
    first_sentence: int
    last_sentence: int


def passage_id(filer_cik: str, text: str) -> str:
    return hashlib.sha1(f"{filer_cik}\n{text}".encode()).hexdigest()[:16]


def _alias_pattern(aliases: Iterable[str]) -> re.Pattern | None:
    words = sorted({a for a in aliases if a}, key=len, reverse=True)
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")


def is_self(name: str, self_names: Iterable[str]) -> bool:
    """Does `name` refer to the filer? Also catches greedy matches such as "Overview Broadcom
    Inc." in Broadcom's own filing."""
    from supplygraph.er.normalize import normalize

    norm = normalize(name)
    if not norm:
        return True
    return any(norm == s or norm.endswith(" " + s) for s in (normalize(n) for n in self_names) if s)


def is_candidate(
    sentence: str, alias_re: re.Pattern | None = None, self_names: Iterable[str] = ()
) -> bool:
    if not SUPPLY_CUE.search(sentence):
        return False
    if any(not is_self(n, self_names) for n in org_names(sentence)):
        return True
    if alias_re is None:
        return False
    return any(not is_self(m.group(0), self_names) for m in alias_re.finditer(sentence))


def aliases_excluding_filer(aliases: Mapping[str, str], self_names: Iterable[str]) -> list[str]:
    self_names = list(self_names)
    return [a for a, name in aliases.items()
            if not is_self(a, self_names) and not is_self(name, self_names)]


def candidate_passages(
    filer_cik: str,
    text: str,
    aliases: Iterable[str] = (),
    *,
    self_names: Iterable[str] = (),
    before: int = 3,
    after: int = 1,
    max_sentence_chars: int = 2000,
    max_passage_chars: int = 6000,
) -> list[Passage]:
    """Windows of sentences around every sentence that both mentions a supply cue and names a
    company. Context before the hit matters: a bulleted list of "suppliers of GPUs such as AMD"
    is only recognisably a *competitor* list if "Our competitors include:" is in view."""
    alias_re = _alias_pattern(aliases)
    self_names = list(self_names)
    sentences = [s[:max_sentence_chars] for s in split_sentences(text)]
    hits = [i for i, s in enumerate(sentences) if is_candidate(s, alias_re, self_names)]
    windows: list[list[int]] = []
    for i in hits:
        lo, hi = max(0, i - before), min(len(sentences) - 1, i + after)
        size = sum(len(s) for s in sentences[windows[-1][0]:hi + 1]) if windows else 0
        if windows and lo <= windows[-1][1] + 1 and size <= max_passage_chars:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    passages = []
    for lo, hi in windows:
        body = " ".join(sentences[lo:hi + 1])
        passages.append(Passage(passage_id(filer_cik, body), filer_cik, body, lo, hi))
    return passages
