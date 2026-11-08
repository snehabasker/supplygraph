"""Name normalization for matching.

"публичное акционерное общество «Нефтяная компания «Роснефть»"  ->  "neftyanaya rosneft"
"PJSC Rosneft Oil Company"                                      ->  "rosneft oil"

Steps: Cyrillic -> Latin transliteration, accent stripping, lowercasing, punctuation removal,
and dropping legal-form / filler words that carry no identity ("pjsc", "gmbh", "company").
"""

from __future__ import annotations

import re
import unicodedata

# Simplified BGN/PCGN-style romanization (Russian, Ukrainian, Belarusian letters).
_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya", "і": "i", "ї": "yi", "є": "ye", "ґ": "g", "ў": "u",
}
_TRANSLIT_TABLE = str.maketrans(_CYRILLIC)

# Legal forms and filler words, as they look *after* transliteration and lowercasing.
LEGAL_FORM_TOKENS = frozenset({
    # English / international
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd",
    "limited", "llc", "llp", "lp", "plc", "public", "joint", "stock", "open", "closed",
    "private", "holding", "holdings", "group", "liability", "responsibility", "partnership",
    "unitary", "enterprise", "the", "of", "and",
    # European abbreviations
    "gmbh", "mbh", "ag", "kg", "se", "sa", "sas", "sarl", "srl", "spa", "nv", "bv", "oy",
    "ab", "as", "asa", "kft", "sro", "zrt", "doo",
    # European long forms (accents already stripped): without these, two unrelated German
    # companies share the token "aktiengesellschaft" and look similar.
    "aktiengesellschaft", "gesellschaft", "mit", "beschrankter", "haftung",
    "kommanditgesellschaft", "societe", "anonyme", "sociedad", "anonima", "limitada",
    "naamloze", "besloten", "vennootschap", "aksjeselskap", "aktiebolag", "osakeyhtio",
    "spolka", "akcyjna", "ograniczona", "odpowiedzialnoscia",
    # Russian / Ukrainian (abbreviations and transliterated long forms)
    "jsc", "pjsc", "ojsc", "cjsc", "pao", "oao", "zao", "ooo", "ao", "tov", "pat",
    "publichnoe", "nepublichnoe", "aktsionernoe", "obshchestvo", "otkrytoe", "zakrytoe",
    "ogranichennoi", "otvetstvennostyu", "kompaniya", "tovarystvo", "unitarnoe", "predpriyatie",
    # Gulf free zones
    "dmcc", "fze", "fzco", "fzc", "fzllc", "est", "wll",
})

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def transliterate(text: str) -> str:
    return text.lower().translate(_TRANSLIT_TABLE)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokens(name: str | None, *, drop_legal_forms: bool = True, translit: bool = True) -> list[str]:
    if not name:
        return []
    text = transliterate(name) if translit else name.lower()
    text = _NON_ALNUM.sub(" ", strip_accents(text))
    out = [t for t in text.split() if len(t) > 1]
    if drop_legal_forms:
        out = [t for t in out if t not in LEGAL_FORM_TOKENS]
    return out


def normalize(name: str | None, *, drop_legal_forms: bool = True, translit: bool = True) -> str:
    return " ".join(tokens(name, drop_legal_forms=drop_legal_forms, translit=translit))
