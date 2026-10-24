#!/usr/bin/env python3
"""Unit tests for the parts of whisper_worker.py that need no model.

    cd Python && ~/Library/Application\\ Support/WhisperHold/runtime/.venv/bin/python3 -m unittest test_whisper_worker
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import whisper_worker  # noqa: E402

SHA256 = "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95"
XET_HASH = "f" * 64


class ContentAddressTests(unittest.TestCase):
    """huggingface-hub names a blob after its SHA-256; since 2.x that blob may be
    a further symlink into a shared store named by Xet hashes, which are also
    64 hex characters but not the SHA-256."""

    def test_plain_cache_layout(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / "blobs").mkdir()
            (root_path / "blobs" / SHA256).write_bytes(b"x")
            snapshot = root_path / "snapshots" / "rev"
            snapshot.mkdir(parents=True)
            os.symlink(f"../../blobs/{SHA256}", snapshot / "model.safetensors")
            self.assertEqual(whisper_worker.content_address(snapshot / "model.safetensors"), SHA256)

    def test_shared_blob_store_layout(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            store = root_path / "shared_blobs" / XET_HASH[:2]
            store.mkdir(parents=True)
            (store / XET_HASH).write_bytes(b"x")
            (root_path / "blobs").mkdir()
            os.symlink(f"../shared_blobs/{XET_HASH[:2]}/{XET_HASH}", root_path / "blobs" / SHA256)
            snapshot = root_path / "snapshots" / "rev"
            snapshot.mkdir(parents=True)
            os.symlink(f"../../blobs/{SHA256}", snapshot / "model.safetensors")
            weights = snapshot / "model.safetensors"
            self.assertEqual(weights.resolve().name, XET_HASH)  # what the old code used
            self.assertEqual(whisper_worker.content_address(weights), SHA256)

    def test_plain_file_has_no_address(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            weights = Path(root) / "model.safetensors"
            weights.write_bytes(b"x")
            self.assertIsNone(whisper_worker.content_address(weights))


class FakeProcessor:
    """Segments carry token ids; the fake tokenizer turns a list of word
    strings straight back into text."""

    class tokenizer:  # noqa: N801 - mirrors the real attribute name
        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            return " ".join(tokens)


class SwallowedWindowTests(unittest.TestCase):
    def test_long_nearly_empty_window_is_reported(self) -> None:
        segments = [
            {"start": 0.0, "end": 4.0, "tokens": "Gerçekten çok teşekkür ederim Erkan abi.".split()},
            {"start": 29.0, "end": 58.98, "tokens": "Bu videoyu izlediğiniz için teşekkür ederim.".split()},
        ]
        self.assertIsNotNone(whisper_worker.swallowed_window(segments, FakeProcessor()))

    def test_normal_speech_rate_is_not_reported(self) -> None:
        segments = [
            {"start": 0.0, "end": 16.0, "tokens": ["kelime"] * 40},
            {"start": 16.0, "end": None, "tokens": ["açık", "uçlu", "son", "parça"]},
        ]
        self.assertIsNone(whisper_worker.swallowed_window(segments, FakeProcessor()))
        self.assertIsNone(whisper_worker.swallowed_window([], FakeProcessor()))


if __name__ == "__main__":
    unittest.main()
