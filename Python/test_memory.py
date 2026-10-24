#!/usr/bin/env python3
"""Unit tests for memory.py with a temporary database and a fake embedder (no model)."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import memory  # noqa: E402


class FakeEmbedder:
    """Bag-of-words vectors over a fixed tiny vocabulary, unit length."""

    vocabulary = ["anahtar", "openclaw", "claude", "responsiv", "sayfa", "mobil", "toplanti", "komite"]

    def encode(self, texts, kind="passage"):
        out = []
        for text in texts:
            words = memory.fold(text).split()
            vector = [float(sum(1 for w in words if token in w)) for token in self.vocabulary]
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / norm for v in vector])
        return out


def count_tokens(text: str) -> int:
    return len(text.split())


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.memory = memory.Memory(Path(self.dir.name) / "memory.sqlite", embedder_factory=FakeEmbedder)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_remember_counts_terms_including_suffixed_forms(self) -> None:
        self.memory.remember_dictation("OpenClaw'ın anahtarı OpenClaw'da kalsın, Claude'nin anahtarı Claude'de.", "app", ["OpenClaw", "Claude", "backend"])
        self.memory.remember_dictation("Backendci arkadaşım geldi.", "app", ["OpenClaw", "backend"])
        stats = self.memory.stats()
        self.assertEqual(stats["dictations"], 2)
        usage = {row["term"]: row["count"] for row in self.memory.db.execute("SELECT * FROM term_usage")}
        self.assertEqual(usage, {"OpenClaw": 1, "Claude": 1, "backend": 1})

    def test_prompt_composition_prefers_used_and_relevant_terms_within_budget(self) -> None:
        self.memory.remember_dictation("OpenClaw'ın anahtarı OpenClaw'da kalsın.", "app", ["OpenClaw"])
        learned = ["Kubernetes", "OpenClaw", "Claude", "Terraform"]
        prompt, chosen = self.memory.compose_prompt("Bağlam.", ["API"], learned, count_tokens, budget=14)
        self.assertEqual(chosen[0], "OpenClaw")
        self.assertTrue(prompt.startswith("Bağlam. Özel adlar ve teknik terimler aynen şu yazımla kullanılır: API, OpenClaw"))
        self.assertLessEqual(count_tokens(prompt), 14)
        self.assertLess(len(chosen), len(learned))

    def test_prompt_without_terms_is_just_the_context(self) -> None:
        prompt, chosen = self.memory.compose_prompt("Bağlam.", [], [], count_tokens)
        self.assertEqual((prompt, chosen), ("Bağlam.", []))

    def test_nearest_corrections_by_embedding_then_fallback(self) -> None:
        self.memory.remember_correction("bekleyen", "backendci", "Dokümanı bekleyen arkadaşım verdi.")
        self.memory.remember_correction("Cloudy'nin", "Claude'nin", "Open Cloud'ın anahtarı, Cloudy'nin anahtarı.")
        self.memory.remember_correction("responsif", "responsive", "Sayfa mobil responsif olacak.")
        # No embedder yet: full-text fallback on shared words.
        hits = self.memory.nearest_corrections("Anahtarı Claude'de bırak.", limit=1)
        self.assertEqual(hits[0]["right"], "Claude'nin")
        # Embedder arrives: vectors are backfilled and used.
        self.memory.start_embedder()
        self.memory.thread.join()
        self.assertEqual(self.memory.stats()["embedded"], 3)
        hits = self.memory.nearest_corrections("Sayfa mobil olmalı.", limit=2)
        self.assertEqual(hits[0]["right"], "responsive")

    def test_forget_and_clear(self) -> None:
        self.memory.remember_correction("bekleyen", "backendci", "Dokümanı bekleyen arkadaşım verdi.")
        self.memory.forget_correction("bekleyen", "backendci")
        self.assertEqual(self.memory.stats()["corrections"], 0)
        self.memory.remember_dictation("Merhaba dünya.", "app", [])
        self.memory.clear()
        self.assertEqual(self.memory.stats()["dictations"], 0)

    def test_term_occurs(self) -> None:
        self.assertTrue(memory.term_occurs("OpenClaw", "OpenClaw'ın anahtarı"))
        self.assertTrue(memory.term_occurs("backend", "backendci geldi"))
        self.assertFalse(memory.term_occurs("backend", "backendcilerinbir"))
        self.assertFalse(memory.term_occurs("claude", "cloudy"))


if __name__ == "__main__":
    unittest.main()
