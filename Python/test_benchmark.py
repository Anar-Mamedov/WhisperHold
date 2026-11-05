#!/usr/bin/env python3
"""benchmark.py builds its set from the memory and a verified set, and counts word errors (no model)."""
import json
import tempfile
import unittest
from pathlib import Path

import benchmark
import memory as memory_store


class FakeModel:
    """Answers every prompt with one fixed reply and keeps the messages it was asked."""

    def __init__(self, answer: str):
        self.answer = answer
        self.asked: list[list[dict[str, str]]] = []

    def answers(self, messages):
        self.asked.append(messages)
        return [self.answer]


class NoExamples:
    def nearest_corrections(self, text):
        return []


class BenchmarkTests(unittest.TestCase):
    def test_word_errors_and_apply(self):
        self.assertEqual(benchmark.word_errors("Ya da bunlar dışında", "Yada bunlar dışında"), 2)
        self.assertEqual(benchmark.word_errors("aynı metin.", "Aynı metin"), 0)
        self.assertEqual(benchmark.apply("Kayıt almaya çalışırsan böyle", [{"wrong": "çalışırsan", "right": "çalışırken"}]),
                         "Kayıt almaya çalışırken böyle")

    def test_set_from_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            store = memory_store.Memory(Path(folder) / "memory.sqlite", embedder_factory=lambda: None)
            store.remember_dictation("Komite toplantısı yarına kaldı, raporu o zaman sunarız.")
            store.remember_dictation("Kayıt almaya çalışırsan böyle bir sorunla karşılaştım.")
            store.remember_dictation("Kısa.")
            store.remember_correction("çalışırsan", "çalışırken", "Kayıt almaya çalışırsan böyle bir sorunla karşılaştım.")
            items = benchmark.set_from_memory(store)
        positives = [it for it in items if it["raw"] != it["expected"]]
        negatives = [it for it in items if it["raw"] == it["expected"]]
        self.assertEqual([it["expected"] for it in positives], ["Kayıt almaya çalışırken böyle bir sorunla karşılaştım."])
        self.assertEqual([it["raw"] for it in negatives], ["Komite toplantısı yarına kaldı, raporu o zaman sunarız."])

    def test_verified_cases_replace_the_memorys_guesses(self):
        unnoticed = "Bu tabloyu bir de dinzeler misin, toplamı tutmuyor."
        corrected = "Liste sayfasında Select Packs sayfadan açılıyor, bunu düzeltelim."
        verified = [
            {"raw": unnoticed, "expected": "Bu tabloyu bir de inceler misin, toplamı tutmuyor."},
            # The memory keeps the text after smart correction; the verified case the first pass.
            {"raw": "Liste sayfasında Select Packs saytan açılıyor, bunu düzeltelim.",
             "expected": "Liste sayfasında Select box zaten açılıyor, bunu düzeltelim.",
             "alternates": [{"wrong": "Packs", "right": "box"}]},
        ]
        with tempfile.TemporaryDirectory() as folder:
            store = memory_store.Memory(Path(folder) / "memory.sqlite", embedder_factory=lambda: None)
            store.remember_dictation("Komite toplantısı yarına kaldı, raporu o zaman sunarız.")
            store.remember_dictation(unnoticed)
            store.remember_dictation(corrected)
            items = benchmark.with_verified(benchmark.set_from_memory(store), verified)
        self.assertEqual(items[:2], verified)
        self.assertEqual([it["raw"] for it in items[2:]], ["Komite toplantısı yarına kaldı, raporu o zaman sunarız."])

    def test_same_dictation(self):
        self.assertTrue(benchmark.same_dictation("Select Packs sayfadan açılıyor, bunu düzeltelim.",
                                                 "Select Packs saytan açılıyor, bunu düzeltelim."))
        self.assertFalse(benchmark.same_dictation("Komite toplantısı yarına kaldı.", "Kayıt almaya çalışırsan böyle."))
        self.assertFalse(benchmark.same_dictation("", ""))

    def test_load_cases_checks_the_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            good = Path(folder) / "good.json"
            good.write_text(json.dumps([{"raw": "a b c", "expected": "a b d", "alternates": [{"wrong": "c", "right": "d"}]}]),
                            encoding="utf-8")
            self.assertEqual(len(benchmark.load_cases(good)), 1)
            for bad in ({"raw": "a"}, [{"raw": "a"}], [{"raw": "a", "expected": "a", "alternates": "c → d"}]):
                path = Path(folder) / "bad.json"
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    benchmark.load_cases(path)

    def test_alternates_reach_the_model_and_the_gate(self):
        case = {"raw": "Liste sayfasında Select Packs açılıyor.", "expected": "Liste sayfasında Select box açılıyor.",
                "alternates": [{"wrong": "Packs", "right": "box"}]}
        model = FakeModel('[{"wrong": "Packs", "right": "box"}]')
        _, applied = benchmark.run_model(case, case["raw"], model, NoExamples(), [], [], ["box"])
        self.assertIn('heard "Packs" as "box"', model.asked[0][-1]["content"])
        self.assertEqual(applied, [{"wrong": "Packs", "right": "box"}])
        # Without the second listening the pair does not sound alike, so the gate stops it.
        _, applied = benchmark.run_model({**case, "alternates": []}, case["raw"], model, NoExamples(), [], [], ["box"])
        self.assertEqual(applied, [])

    def test_leave_one_out_covers_a_whole_dictation(self):
        corrections = [
            {"wrong": "çalışırsan", "right": "çalışırken", "context": "Kayıt almaya çalışırsan böyle bir sorunla karşılaştım."},
            {"wrong": "stoneları", "right": "sütunları", "context": "Tablonun stoneları kaymış."},
        ]
        raw = "Dün denedim. Kayıt almaya çalışırsan böyle bir sorunla karşılaştım. Sonra kapattım."
        self.assertEqual(benchmark.taught_on(raw, corrections), {benchmark.term_llm.fold("çalışırsan")})

        # "Packs" → "box" does not sound alike: the gate lets it through only as a taught pair.
        raw = "Dün denedim. Select Packs açılıyor. Sonra kapattım."
        case = {"raw": raw, "expected": raw.replace("Packs", "box")}
        model = FakeModel('[{"wrong": "Packs", "right": "box"}]')
        learned = [("Packs", "box")]
        _, applied = benchmark.run_model(case, raw, model, NoExamples(), [], learned, [])
        self.assertEqual(applied, [{"wrong": "Packs", "right": "box"}])
        # Taught on this very dictation, so the model has to find it without that help.
        taught_here = [{"wrong": "Packs", "right": "box", "context": "Select Packs açılıyor."}]
        _, applied = benchmark.run_model(case, raw, model, NoExamples(), taught_here, learned, [])
        self.assertEqual(applied, [])


if __name__ == "__main__":
    unittest.main()
