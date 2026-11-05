#!/usr/bin/env python3
"""benchmark_audio.py: settings as the app derives them, the memory as it was, WAV reading (no model)."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import benchmark
import benchmark_audio
import memory as memory_store

SWIFT = '''
    static let sampleContextPrompt =
        "Qwen ve Whisper modellerini " +
        "konuşuyoruz."

    static let sampleTechnicalTerms =
        "Qwen, build, backendci"

    @Published var language: String
'''


def wav_bytes(samples: bytes, tag: int, bits: int, rate: int = 16_000, channels: int = 1) -> bytes:
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


class SettingsTests(unittest.TestCase):
    def test_defaults_come_from_the_swift_source(self):
        self.assertEqual(benchmark_audio.swift_sample("sampleContextPrompt", SWIFT), "Qwen ve Whisper modellerini konuşuyoruz.")
        settings = benchmark_audio.read_settings(stored={}, source=SWIFT)
        self.assertEqual(settings["contexts"], ["Qwen ve Whisper modellerini konuşuyoruz."])
        self.assertEqual(settings["terms"], ["Qwen", "build", "backendci"])
        self.assertEqual(settings["language"], "tr")
        self.assertTrue(settings["builtin"])

    def test_learned_corrections_and_terms(self):
        stored = {"vocabulary": "backendci, Swift", "learnedCorrections": json.dumps([
            {"wrong": "dosya", "right": "dosyaya", "context": "Bu dosya kaydet.", "learnedAt": 100.0},
            {"wrong": "bekent çi", "right": "backendçi", "context": "bekent çi yaptı", "learnedAt": 50.0},
            {"wrong": "Dosyaya", "right": "DOSYAYA", "context": "x", "learnedAt": 10.0},
        ]).encode()}
        settings = benchmark_audio.read_settings(stored=stored, source=SWIFT)
        moment = benchmark_audio.APPLE_EPOCH + 60
        before = benchmark_audio.learned_before(settings, moment)
        self.assertEqual([c["right"] for c in before], ["backendçi", "DOSYAYA"])
        # "backendçi" folds onto the listed "backendci"; the two spellings of "dosyaya" count once.
        self.assertEqual(benchmark_audio.learned_terms(settings["terms"], settings["learned"]), ["dosyaya"])
        self.assertEqual(benchmark_audio.learned_terms(settings["terms"], before), ["DOSYAYA"])

    def test_recorded_at(self):
        moment = benchmark_audio.recorded_at("dikte-2026-10-28T14-05-29Z.wav")
        self.assertEqual(moment, 1793196329.0)
        self.assertIsNone(benchmark_audio.recorded_at("kayit.wav"))


class AudioTests(unittest.TestCase):
    def test_read_wav_float_and_pcm(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.wav"
            path.write_bytes(wav_bytes(struct.pack("<3f", 0.5, -0.25, 0.0), tag=3, bits=32))
            self.assertEqual(benchmark_audio.read_wav(path).tolist(), [0.5, -0.25, 0.0])
            path.write_bytes(wav_bytes(struct.pack("<2h", 16384, -32768), tag=1, bits=16))
            self.assertEqual(benchmark_audio.read_wav(path).tolist(), [0.5, -1.0])
            path.write_bytes(wav_bytes(struct.pack("<4f", 1.0, 0.0, 0.5, 0.5), tag=3, bits=32, channels=2))
            self.assertEqual(benchmark_audio.read_wav(path).tolist(), [0.5, 0.5])
            path.write_bytes(wav_bytes(struct.pack("<1f", 0.1), tag=3, bits=32, rate=44_100))
            with self.assertRaises(ValueError):
                benchmark_audio.read_wav(path)

    def test_hint_effects(self):
        effects = benchmark_audio.hint_effects(
            "Ayarlar menüs açılıyor", [{"wrong": "menüs", "right": "menüsü"}, {"wrong": "açılıyor", "right": "kapanıyor"},
                                       {"wrong": "Ayarlar", "right": "ayarlar"}],
            "Ayarlar menüsü açılıyor")
        self.assertEqual(effects, {"doğru": 1, "yanlış": 1, "nötr": 1})


class MemoryTests(unittest.TestCase):
    def test_memory_as_of_drops_what_came_later_and_leaves_the_original(self):
        with tempfile.TemporaryDirectory() as folder:
            support = Path(folder) / "support"
            original = memory_store.Memory(support / "memory.sqlite", embedder_factory=lambda: None)
            original.remember_dictation("Eski bir dikte burada duruyor.")
            original.remember_dictation("Yeni bir dikte sonradan geldi.")
            original.remember_correction("dosya", "dosyayı", "Bu dosya kaydet.")
            with original.lock:
                original.db.execute("UPDATE dictations SET created_at = 100 WHERE text LIKE 'Eski%'")
                original.db.execute("UPDATE dictations SET created_at = 300 WHERE text LIKE 'Yeni%'")
                original.db.execute("UPDATE corrections SET created_at = 300")
                original.db.commit()
            saved = benchmark.SUPPORT
            benchmark.SUPPORT = support
            try:
                copy = benchmark_audio.memory_as_of(200, Path(folder), embedder=None)
            finally:
                benchmark.SUPPORT = saved
            self.assertEqual(copy.recent_dictations(), ["Eski bir dikte burada duruyor."])
            self.assertEqual(copy.learned_pairs(), [])
            copy.db.close()
            self.assertEqual(len(original.recent_dictations()), 2)
            self.assertEqual(original.learned_pairs(), [("dosya", "dosyayı")])


if __name__ == "__main__":
    unittest.main()
