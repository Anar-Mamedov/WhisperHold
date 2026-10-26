"""Spelled-out acronyms: "el eleme gönderdim" → "LLM'e gönderdim".

Whisper, decoding Turkish, writes an acronym the speaker spells letter by
letter as the Turkish (or English) names of those letters: "LLM'e" comes out
as "el eleme", "API" as "ey pi ay". No prompt or biasing fixes this reliably,
because the model heard exactly what it wrote — syllables that are real
words. This module reverses it deterministically: for every acronym the user
listed (or the bundled vocabulary carries), a run of two or more words whose
letters spell the acronym's letter names, optionally followed by a Turkish
suffix, becomes the acronym with the suffix after an apostrophe.

Only windows of at least two words are touched, and the window has to
spell the whole acronym; "el" alone, or "eleme" alone, is left as it is.
"""

from __future__ import annotations

import re
from typing import Iterable

# Letter names a Turkish speaker uses for an acronym: the Turkish name first,
# then the English one as it is pronounced (and written back by Whisper).
LETTER_NAMES: dict[str, list[str]] = {
    "a": ["a", "ey"], "b": ["be", "bi"], "c": ["ce", "si"], "ç": ["çe"], "d": ["de", "di"],
    "e": ["e", "i"], "f": ["fe", "ef"], "g": ["ge", "ci"], "ğ": ["yumuşakge"], "h": ["he", "aş", "eyç", "haş"],
    "i": ["i", "ay"], "j": ["je", "cey"], "k": ["ke", "ka", "key"], "l": ["le", "el"], "m": ["me", "em"],
    "n": ["ne", "en"], "o": ["o", "ov"], "ö": ["ö"], "p": ["pe", "pi"], "q": ["kü", "kyu", "kju"],
    "r": ["re", "er", "ar"], "s": ["se", "es"], "ş": ["şe"], "t": ["te", "ti"], "u": ["u", "yu"], "ü": ["ü"],
    "v": ["ve", "vi"], "w": ["çiftve", "dabılyu", "dabilyu", "dablyu"], "x": ["iks", "eks"],
    "y": ["ye", "vay"], "z": ["ze", "zet", "zi"],
}

# The Turkish suffixes an acronym takes ("LLM'e", "API'yi", "GPU'da").
SUFFIX = re.compile(
    r"^(?:ye|ya|yi|yu|yı|yü|e|a|i|u|ı|ü|de|da|te|ta|den|dan|ten|tan|nin|nın|nun|nün|in|ın|un|ün|"
    r"le|la|yle|yla|ler|lar|leri|ları|lerin|ların|lere|lara|lerde|larda|lerden|lardan|ye|ya|"
    r"yi|yı|yu|yü|si|sı|su|sü|ci|cı|cu|cü|ler|lar|dir|dır|dur|dür|tir|tır|tur|tür|"
    r"ler|lar|e|a|i|ı|u|ü)+$"
)
ACRONYM = re.compile(r"^[A-ZÇĞİÖŞÜ]{2,6}$")
TOKEN = re.compile(r"\S+")
EDGE = re.compile(r"^(?P<lead>[^\w]*)(?P<core>.*?)(?P<tail>[^\w]*)$", re.DOTALL)
MAX_WINDOW = 7


def fold(text: str) -> str:
    return text.lower().replace("ı", "i").replace("i̇", "i").replace("'", "").replace("’", "")


def is_acronym(term: str) -> bool:
    return ACRONYM.match(term.strip()) is not None


def pattern_for(acronym: str) -> re.Pattern[str] | None:
    parts = []
    for letter in fold(acronym):
        names = LETTER_NAMES.get(letter)
        if not names:
            return None
        parts.append("(?:" + "|".join(sorted(names, key=len, reverse=True)) + ")")
    return re.compile("^" + "".join(parts) + r"(?P<suffix>[a-zçğıiöşü]{0,6})$")


def restore(text: str, acronyms: Iterable[str]) -> str:
    """Replace spelled-out runs of words with the acronym they spell."""
    patterns = [(acronym.strip(), pattern) for acronym in dict.fromkeys(a.strip() for a in acronyms if is_acronym(a))
                if (pattern := pattern_for(acronym.strip())) is not None]
    if not patterns:
        return text
    tokens = [(m.start(), m.end(), m.group(0)) for m in TOKEN.finditer(text)]
    pieces: list[str] = []
    cursor = 0
    index = 0
    while index < len(tokens):
        replacement = None
        for size in range(min(MAX_WINDOW, len(tokens) - index), 1, -1):
            window = tokens[index:index + size]
            cores = [EDGE.match(token).groupdict() for _, _, token in window]
            # Punctuation between the words ends the run ("el, eleme" is two things).
            if any(c["tail"] for c in cores[:-1]) or any(c["lead"] for c in cores[1:]):
                continue
            if any(not c["core"].replace("'", "").replace("’", "").isalpha() for c in cores):
                continue
            joined = fold("".join(c["core"] for c in cores))
            for acronym, pattern in patterns:
                match = pattern.match(joined)
                if match is None:
                    continue
                suffix = match.group("suffix")
                if suffix and SUFFIX.match(suffix) is None:
                    continue
                last = cores[-1]["core"].replace("'", "").replace("’", "")
                original_suffix = last[len(last) - len(suffix):] if suffix else ""
                if fold(original_suffix) != suffix:
                    continue
                replacement = (window[0][0], window[-1][1],
                               cores[0]["lead"] + acronym + ("'" + original_suffix if suffix else "") + cores[-1]["tail"])
                break
            if replacement:
                break
        if replacement:
            start, end, new = replacement
            pieces.append(text[cursor:start]); pieces.append(new)
            cursor = end
            index += size
        else:
            index += 1
    pieces.append(text[cursor:])
    return "".join(pieces)
