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
    def test_term_replacements_within_the_sound_budget_are_kept(self) -> None:
        text = "Onu be kentçi arkadaşım verdi, komit attım, paypline'ı kırdım, RVD modeli."
        pairs = [
            {"wrong": "be kentçi", "right": "backendci"},
            {"wrong": "komit", "right": "commit"},
            {"wrong": "paypline'ı", "right": "pipeline'ı"},
            {"wrong": "RVD", "right": "RWD"},
        ]
        self.assertEqual(term_llm.validate(pairs, text, VOCAB + ["backendci", "RWD"]), pairs)

    def test_far_sounds_wrong_suffixes_and_absent_sources_are_dropped(self) -> None:
        text = "Bekleyen arkadaşım geldi, Giral zaten, e-postada yazıyor."
        pairs = [
            {"wrong": "arkadaşım", "right": "backend"},      # in vocabulary, but sounds nothing alike
            {"wrong": "geldi", "right": "GitHubçıların"},    # too long a suffix (6 letters), not a word
            {"wrong": "gitti", "right": "GitHub"},           # not in the text
            {"wrong": "Giral", "right": "Git"},              # 3 of 5 letters: beyond the budget
            {"wrong": "e-postada", "right": "endpoint'te"},  # a 12B model really proposed this
            {"wrong": "geldi", "right": "geldi."},           # sentence punctuation
            {"wrong": "", "right": "API"},
            "junk",
        ]
        self.assertEqual(term_llm.validate(pairs, text, VOCAB + ["Git"]), [])

    def test_learned_pairs_pass_without_further_checks(self) -> None:
        text = "Bu dokümanı bekleyen arkadaşım verdi."
        pair = {"wrong": "bekleyen", "right": "backendci"}   # six edits: only a taught pair gets through
        self.assertEqual(term_llm.validate([pair], text, VOCAB), [])
        self.assertEqual(term_llm.validate([pair], text, VOCAB, [("Bekleyen", "backendci")]), [pair])

    def test_dictionary_words_are_accepted_only_when_the_dictionary_is_there(self) -> None:
        import dictionary

        text = "Kayıt almaya çalışırsan böyle bir sorunla karşılaştım."
        pair = {"wrong": "çalışırsan", "right": "çalışırken"}
        original = dictionary.is_word
        try:
            dictionary.is_word = lambda word, languages=None: None     # no PyObjC: terms only
            self.assertEqual(term_llm.validate([pair], text, VOCAB), [])
            dictionary.is_word = lambda word, languages=None: True     # the macOS dictionary knows the word
            self.assertEqual(term_llm.validate([pair], text, VOCAB), [pair])
            dictionary.is_word = lambda word, languages=None: False
            self.assertEqual(term_llm.validate([pair], text, VOCAB), [])
        finally:
            dictionary.is_word = original

    def test_words_that_already_are_terms_are_left_alone(self) -> None:
        text = "Bu dokümanı backendci arkadaşım verdi, commit attım."
        pairs = [{"wrong": "backendci", "right": "backend"}, {"wrong": "commit", "right": "committen"}]
        self.assertEqual(term_llm.validate(pairs, text, VOCAB), [])

    def test_english_inflection_is_not_a_turkish_suffix(self) -> None:
        text = "Batonlar bozuldu, paypline kırıldı."
        pairs = [{"wrong": "Batonlar", "right": "buttons"}, {"wrong": "paypline", "right": "pipeline'ı"}]
        self.assertEqual(term_llm.validate(pairs, text, ["button", "pipeline"]), [{"wrong": "paypline", "right": "pipeline'ı"}])

    def test_sounds_alike(self) -> None:
        self.assertTrue(term_llm.sounds_alike("be kentçi", "backendci"))
        self.assertTrue(term_llm.sounds_alike("Cloudy", "Claude"))
        self.assertTrue(term_llm.sounds_alike("Open Cloud", "OpenClaw"))
        self.assertTrue(term_llm.sounds_alike("çıktığının", "çıktının"))
        self.assertFalse(term_llm.sounds_alike("bekleyen", "backendci"))
        self.assertTrue(term_llm.sounds_alike("paypline", "pipeline'ı"))
        self.assertTrue(term_llm.sounds_alike("komit", "commit"))
        self.assertFalse(term_llm.sounds_alike("Giral", "Git"))
        self.assertFalse(term_llm.sounds_alike("grep", "deploy"))
        self.assertFalse(term_llm.sounds_alike("a", "API çağrısı yapan servis"))
        self.assertEqual(term_llm.sound("backendci"), "bakendci")
        self.assertEqual(term_llm.sound("commit"), "kommit")

    def test_identity_and_limit(self) -> None:
        text = " ".join(f"a{i}" for i in range(20))
        pairs = [{"wrong": f"a{i}", "right": "a0"} for i in range(1, 20)]
        self.assertEqual(len(term_llm.validate(pairs, text, ["a0"])), term_llm.MAX_REPLACEMENTS)
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

    def test_proposals_of_several_models_merge_first_wins(self) -> None:
        first = [{"wrong": "yaptınca", "right": "yaptın ki"}, {"wrong": "RVD", "right": "RWD"}]
        second = [{"wrong": "yaptınca", "right": "yapınca"}, {"wrong": "çıktığının", "right": "çıktının"}, "junk"]
        merged = term_llm.merge_proposals([first, second])
        self.assertEqual([(m["wrong"], m["right"], m["model"]) for m in merged],
                         [("yaptınca", "yaptın ki", 0), ("RVD", "RWD", 0), ("çıktığının", "çıktının", 1)])
        self.assertEqual(term_llm.merge_proposals([[], []]), [])

    def test_a_single_turkish_word_never_becomes_a_term_unless_taught(self) -> None:
        import dictionary

        turkish = {"komite", "toplantısı", "yarına", "kaldı", "çalışırsan", "çalışırken", "front", "ent"}
        original_word, original_turkish = dictionary.is_word, dictionary.is_turkish_word
        try:
            dictionary.is_word = lambda word, languages=None: True
            dictionary.is_turkish_word = lambda word: word.strip("'’.,").lower() in turkish
            text = "Komite toplantısı yarına kaldı, çalışırsan haber ver."
            komite = {"wrong": "Komite", "right": "Commit"}
            self.assertEqual(term_llm.validate([{**komite, "model": 1}], text, VOCAB), [])
            self.assertEqual(term_llm.validate([{**komite, "model": 0}], text, VOCAB), [])
            self.assertEqual(term_llm.validate([komite], text, VOCAB), [])
            real = {"wrong": "çalışırsan", "right": "çalışırken"}
            self.assertEqual(term_llm.validate([{**real, "model": 1}], text, VOCAB), [real])
            # A split term whose halves are dictionary words is still a term fix.
            split = {"wrong": "Front ent", "right": "Frontend"}
            self.assertEqual(term_llm.validate([{**split, "model": 1}], "Front ent tarafı bozuk.", VOCAB + ["Frontend"]), [split])
            # A taught pair is never subject to the rule.
            self.assertEqual(term_llm.validate([{**komite, "model": 1}], text, VOCAB, [("Komite", "Commit")]), [komite])
        finally:
            dictionary.is_word, dictionary.is_turkish_word = original_word, original_turkish

    def test_model_list_follows_memory_and_override(self) -> None:
        import os

        original = os.environ.get("WHISPERHOLD_LLM")
        try:
            os.environ["WHISPERHOLD_LLM"] = "a/one, b/two"
            self.assertEqual(term_llm.mlx_model_ids(), ["a/one", "b/two"])
            os.environ.pop("WHISPERHOLD_LLM")
            self.assertEqual(term_llm.mlx_model_ids(), ["mlx-community/gemma-4-12B-it-qat-4bit"])
        finally:
            if original is not None:
                os.environ["WHISPERHOLD_LLM"] = original
            else:
                os.environ.pop("WHISPERHOLD_LLM", None)

    def test_answer_parsing(self) -> None:
        self.assertEqual(term_llm.parse_answer('Sure: [{"wrong": "a", "right": "b"}] done'), [{"wrong": "a", "right": "b"}])
        self.assertEqual(term_llm.parse_answer("[]"), [])
        self.assertEqual(term_llm.parse_answer("no json here"), [])
        self.assertEqual(term_llm.parse_answer("[1, 2"), [])


if __name__ == "__main__":
    unittest.main()


class AlternateTests(unittest.TestCase):
    def test_second_listening_hint_and_gate(self) -> None:
        text = "Erhan Bey, bu ekranda stoneları sıfırla derseniz orası düzelir."
        alternates = [{"wrong": "stoneları", "right": "sütunları"}, {"wrong": "orası", "right": "orası"}]
        messages = term_llm.build_messages(text, [], alternates)
        self.assertTrue(messages[-1]["content"].startswith(text))
        self.assertIn('"stoneları" as "sütunları"', messages[-1]["content"])
        self.assertNotIn("orası", messages[-1]["content"].split("(A second")[1])
        self.assertEqual(term_llm.build_messages(text, [])[-1]["content"], text)
        import dictionary

        original = dictionary.is_word, dictionary.is_turkish_word
        try:
            dictionary.is_word = lambda word, languages=None: True
            dictionary.is_turkish_word = lambda word: word.lower() in {"sütunları", "stoneları"}
            far = {"wrong": "stoneları", "right": "kolonları"}   # not what the second pass heard, and far in sound
            heard = {"wrong": "stoneları", "right": "sütunları"}
            self.assertEqual(term_llm.validate([far], text, VOCAB, None, alternates), [])
            self.assertEqual(term_llm.validate([heard], text, VOCAB, None, alternates), [heard])
        finally:
            dictionary.is_word, dictionary.is_turkish_word = original
