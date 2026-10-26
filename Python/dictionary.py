"""Is this a real word? macOS's own spelling dictionaries, asked through PyObjC.

The Swift side already uses NSSpellChecker (SpellingGuard) to keep the term
corrector away from real words. The worker needs the same answer for the
smart-correction gate: a proposed replacement may only be a word the
dictionary (or the vocabulary) knows. Without PyObjC the answer is "unknown",
and the gate falls back to vocabulary terms only.
"""

from __future__ import annotations

import threading
from functools import lru_cache

LANGUAGES = ("tr", "en")
_lock = threading.Lock()
_checker = None
_unavailable = False


def available() -> bool:
    return _load() is not None


def _load():
    global _checker, _unavailable
    if _checker is not None or _unavailable:
        return _checker
    with _lock:
        if _checker is None and not _unavailable:
            try:
                from AppKit import NSSpellChecker  # type: ignore

                _checker = NSSpellChecker.sharedSpellChecker()
            except Exception:  # noqa: BLE001 - PyObjC missing or AppKit unusable
                _unavailable = True
    return _checker


@lru_cache(maxsize=4096)
def is_word(word: str, languages: tuple[str, ...] = LANGUAGES) -> bool | None:
    """True/False from the dictionaries, None when no dictionary is available."""
    checker = _load()
    if checker is None:
        return None
    cleaned = word.strip("'’.,;:!?\"()[]").lower()
    if not cleaned or not any(ch.isalpha() for ch in cleaned):
        return True
    with _lock:
        for language in languages:
            result = checker.checkSpellingOfString_startingAt_language_wrap_inSpellDocumentWithTag_wordCount_(
                cleaned, 0, language, False, 0, None
            )
            # NSNotFound: nothing misspelled in the string.
            if result[0].location > 10**9:
                return True
    return False


def is_turkish_word(word: str) -> bool | None:
    """The Turkish dictionary alone: "komite" yes, "commit" no."""
    return is_word(word, ("tr",))
