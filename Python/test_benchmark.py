#!/usr/bin/env python3
"""benchmark.py builds its set from the memory and counts word errors (no model)."""
import tempfile
import unittest
from pathlib import Path

import benchmark
import memory as memory_store


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


if __name__ == "__main__":
    unittest.main()
