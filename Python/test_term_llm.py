#!/usr/bin/env python3
"""Unit tests for term_llm.py's validation (no model needed)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import term_llm  # noqa: E402

VOCAB = ["backend", "commit", "endpoint", "pipeline", "GitHub", "API"]


class ValidateTests(unittest.TestCase):
    def test_vocabulary_derived_replacements_are_kept(self) -> None:
        text = "Bu dokümanı bekleyen arkadaşım verdi, komit attım, paypline'ı kırdım."
        pairs = [
            {"wrong": "bekleyen", "right": "backendci"},
            {"wrong": "komit", "right": "commit"},
            {"wrong": "paypline'ı", "right": "pipeline'ı"},
        ]
        self.assertEqual(term_llm.validate(pairs, text, VOCAB), pairs)

    def test_unknown_targets_and_absent_sources_are_dropped(self) -> None:
        text = "Bekleyen arkadaşım geldi."
        pairs = [
            {"wrong": "bekleyen", "right": "beklemeyen"},   # not a term
            {"wrong": "arkadaşım", "right": "backend"},      # in vocabulary, but sounds nothing alike
            {"wrong": "geldi", "right": "GitHubçıların"},    # too long a suffix (6 letters)
            {"wrong": "gitti", "right": "GitHub"},           # not in the text
            {"wrong": "", "right": "API"},
            "junk",
        ]
        self.assertEqual(term_llm.validate(pairs, text, VOCAB), [])

    def test_words_that_already_are_terms_are_left_alone(self) -> None:
        text = "Bu dokümanı backendci arkadaşım verdi, commit attım."
        pairs = [{"wrong": "backendci", "right": "backend"}, {"wrong": "commit", "right": "committen"}]
        self.assertEqual(term_llm.validate(pairs, text, VOCAB), [])

    def test_english_inflection_is_not_a_turkish_suffix(self) -> None:
        text = "Batonlar bozuldu, paypline kırıldı."
        pairs = [{"wrong": "Batonlar", "right": "buttons"}, {"wrong": "paypline", "right": "pipeline'ı"}]
        self.assertEqual(term_llm.validate(pairs, text, ["button", "pipeline"]), [{"wrong": "paypline", "right": "pipeline'ı"}])

    def test_sounds_alike(self) -> None:
        self.assertTrue(term_llm.sounds_alike("bekleyen", "backendci"))
        self.assertTrue(term_llm.sounds_alike("komite", "commit"))
        self.assertTrue(term_llm.sounds_alike("Kuhra kıvresti", "pull request'i") is False)
        self.assertFalse(term_llm.sounds_alike("grep", "deploy"))
        self.assertFalse(term_llm.sounds_alike("a", "API çağrısı yapan servis"))

    def test_identity_and_limit(self) -> None:
        text = " ".join(f"a{i}" for i in range(20))
        pairs = [{"wrong": f"a{i}", "right": "API"} for i in range(20)]
        self.assertEqual(len(term_llm.validate(pairs, text, VOCAB)), term_llm.MAX_REPLACEMENTS)
        self.assertEqual(term_llm.validate([{"wrong": "API", "right": "api"}], "API çağrısı", VOCAB), [])

    def test_user_examples_become_worked_examples(self) -> None:
        examples = [
            {"text": "Bu dokümanı bekleyen arkadaşım verdi.", "wrong": "bekleyen", "right": "backendci"},
            {"text": "Alakasız cümle.", "wrong": "yok", "right": "API"},  # wrong word not in context: skipped
        ]
        messages = term_llm.build_messages("Yeni metin.", examples)
        self.assertEqual(messages[-1], {"role": "user", "content": "Yeni metin."})
        self.assertEqual(messages[-3]["content"], "Bu dokümanı bekleyen arkadaşım verdi.")
        self.assertIn('"backendci"', messages[-2]["content"])
        self.assertEqual(len(messages), 1 + 2 * len(term_llm.FEW_SHOT) + 2 + 1)

    def test_answer_parsing(self) -> None:
        self.assertEqual(term_llm.parse_answer('Sure: [{"wrong": "a", "right": "b"}] done'), [{"wrong": "a", "right": "b"}])
        self.assertEqual(term_llm.parse_answer("[]"), [])
        self.assertEqual(term_llm.parse_answer("no json here"), [])
        self.assertEqual(term_llm.parse_answer("[1, 2"), [])


if __name__ == "__main__":
    unittest.main()
