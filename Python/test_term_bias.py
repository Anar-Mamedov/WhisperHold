#!/usr/bin/env python3
"""Unit tests for term_bias.py with a fake tokenizer (no model needed).

    cd Python && ~/Library/Application\\ Support/WhisperHold/runtime/.venv/bin/python3 -m unittest test_term_bias
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import term_bias  # noqa: E402


class FakeTokenizer:
    """Splits on spaces; each piece is a 'token' whose id is its hash bucket.

    " backendci" tokenizes as [" backend", "ci"] like the real tokenizer would.
    """

    pieces = {" backend": 1, "ci": 2, " Backend": 3, " pull": 4, " request": 5, " bek": 6, "ley": 7, "en": 8, " Pull": 9}

    def encode(self, text, add_special_tokens=False):
        out = []
        rest = text
        while rest:
            for piece, token in sorted(self.pieces.items(), key=lambda kv: -len(kv[0])):
                if rest.startswith(piece):
                    out.append(token)
                    rest = rest[len(piece):]
                    break
            else:
                rest = rest[1:]  # unknown character: skip
        return out


class VocabularyTests(unittest.TestCase):
    def test_bundled_list_loads(self) -> None:
        terms = term_bias.load_vocabulary()
        self.assertGreater(len(terms), 300)
        self.assertIn("backend", terms)
        self.assertIn("backendci", terms)
        self.assertEqual(len(terms), len(set(terms)))

    def test_comments_blanks_and_duplicates_are_dropped(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("# yorum\n\nbackend\n  frontend \nbackend\n")
        self.assertEqual(term_bias.load_vocabulary(Path(handle.name)), ["backend", "frontend"])

    def test_spellings(self) -> None:
        self.assertEqual(term_bias.spellings("backend"), [" backend", " Backend"])
        self.assertEqual(term_bias.spellings("GitHub"), [" GitHub"])


class TrieTests(unittest.TestCase):
    def test_start_tokens_and_continuations(self) -> None:
        trie = term_bias.TermTrie(FakeTokenizer(), ["backendci", "pull request"])
        self.assertEqual(trie.start_tokens, [1, 3, 4, 9])  # " backend", " Backend", " pull", " Pull"
        self.assertEqual(trie.max_depth, 2)
        self.assertEqual(trie.continuations([9, 1]), {2})      # after " backend" → "ci"
        self.assertEqual(trie.continuations([4]), {5})         # after " pull" → " request"
        self.assertEqual(trie.continuations([6, 7]), set())    # "bekley…" is no term prefix
        self.assertEqual(trie.continuations([]), set())


if __name__ == "__main__":
    unittest.main()
