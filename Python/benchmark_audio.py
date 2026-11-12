#!/usr/bin/env python3
"""Measure the whole dictation chain on recordings whose text was checked by listening.

`benchmark.py` measures the text layers on the memory's sentences. This tool
starts from the audio. For every verified case that names a recording in
~/Documents/WhisperHold it runs what the app runs — the prompt composed from
the memory, contextual biasing, the first pass, the second listening, smart
correction — and counts word errors against the verified text after the
first pass and after correction. Run it after changing the prompt, the
biasing, the decoder settings or the second listening.

Each recording is decoded with the memory and the learned corrections as
they were when it was dictated: a copy of memory.sqlite without the
dictations and corrections stored later, and only the corrections learned
before it. A correction taught on that very dictation cannot hand the answer
back. The real memory is never written.

A guard set (benchmark_guard.json: software-term sentences and ordinary
Turkish full of look-alike words, spoken by macOS's Turkish voice) catches a
change that helps the user's recordings but damages ordinary speech or term
spelling. Its audio is synthesized on first use into a temporary folder.

The archive in ~/Documents/WhisperHold keeps only its newest few recordings,
so a verified case would lose its audio within a day. Every run therefore
copies the verified recordings still in the archive into
~/Library/Application Support/WhisperHold/benchmark_recordings, next to the
verified set, and reads them from there afterwards.

Not covered: the Swift post-processing (term corrector, replacements, the
hallucination filter), which is the same for every variant measured here.

    cd Python
    ~/Library/Application\\ Support/WhisperHold/runtime/.venv/bin/python3 benchmark_audio.py
    ... --no-smart          # Whisper only, no smart correction
    ... --save run.json     # keep every transcript for a later comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Library/Application Support/WhisperHold/HuggingFace"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import benchmark  # noqa: E402

ARCHIVE = Path.home() / "Documents/WhisperHold"
# Where verified recordings are kept once the tool has seen them: the archive
# rotates (RecordingArchive.retainCount), this folder does not. In November
# 2026 four of the five recordings verified a week earlier had already been
# rotated out and the set could no longer be measured.
KEPT = benchmark.SUPPORT / "benchmark_recordings"
GUARD = Path(__file__).with_name("benchmark_guard.json")
GUARD_AUDIO = Path(tempfile.gettempdir()) / "whisperhold-guard"
PREFERENCES_SWIFT = Path(__file__).resolve().parent.parent / "Sources/Settings/Preferences.swift"
BUNDLE_ID = "com.anar.whisperhold"
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp()
SAMPLE_RATE = 16_000
SWIFT_EXAMPLES = 4  # AppController.polish sends learnedCorrections.prefix(4)


# MARK: the app's settings

def swift_sample(name: str, source: str) -> str:
    """A `static let <name> = "…" + "…"` default from Preferences.swift, read
    rather than copied so the app and this tool cannot drift apart."""
    match = re.search(rf"static let {name} =(.*?)\n\s*\n", source, re.DOTALL)
    if not match:
        raise ValueError(f"Preferences.swift: {name} bulunamadı")
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1)))


def fold_key(text: str) -> str:
    """LearnedCorrection.key: case- and diacritic-insensitive."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def read_settings(stored: dict[str, Any] | None = None, source: str | None = None) -> dict[str, Any]:
    """The settings the worker receives, as Preferences.swift derives them."""
    if stored is None:
        try:
            raw = subprocess.run(["defaults", "export", BUNDLE_ID, "-"], capture_output=True, check=True).stdout
            stored = plistlib.loads(raw)
        except (subprocess.CalledProcessError, plistlib.InvalidFileException):
            stored = {}
    if source is None:
        source = PREFERENCES_SWIFT.read_text(encoding="utf-8")
    context = stored.get("contextPrompt", swift_sample("sampleContextPrompt", source))
    vocabulary = stored.get("vocabulary", swift_sample("sampleTechnicalTerms", source))
    learned = json.loads(stored["learnedCorrections"]) if stored.get("learnedCorrections") else []
    return {
        "language": stored.get("language", "tr"),
        "contexts": [line.strip() for line in context.splitlines() if line.strip()],
        "terms": [term.strip() for term in re.split(r"[,\n]", vocabulary) if term.strip()],
        "builtin": bool(stored.get("useBuiltInVocabulary", True)),
        "learned": [{**c, "learned_at": APPLE_EPOCH + float(c["learnedAt"])} for c in learned],
    }


def learned_before(settings: dict[str, Any], moment: float | None) -> list[dict[str, Any]]:
    """The learned corrections that existed at `moment` (all of them when None), newest first."""
    return [c for c in settings["learned"] if moment is None or c["learned_at"] < moment]


def learned_terms(terms: list[str], learned: list[dict[str, Any]]) -> list[str]:
    """Preferences.learnedTerms: the corrected spellings not already listed, once each."""
    listed = {fold_key(term) for term in terms}
    seen: set[str] = set()
    out = []
    for correction in learned:
        key = fold_key(correction["right"])
        if key not in listed and key not in seen:
            seen.add(key)
            out.append(correction["right"])
    return out


def recorded_at(name: str) -> float | None:
    """The moment in an archive file name (dikte-2026-10-28T14-05-29Z.wav)."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z", name)
    if not match:
        return None
    return datetime.fromisoformat(f"{match[1]}T{match[2]}:{match[3]}:{match[4]}+00:00").timestamp()


# MARK: memory and audio

def memory_as_of(moment: float | None, folder: Path, embedder: Any) -> Any:
    """A copy of memory.sqlite without what was stored at or after `moment`."""
    import memory as memory_store

    path = folder / f"memory-{int(moment or 0)}.sqlite"
    path.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{benchmark.SUPPORT / 'memory.sqlite'}?mode=ro", uri=True)
    target = sqlite3.connect(str(path))
    source.backup(target)
    source.close()
    target.close()
    store = memory_store.Memory(path, embedder_factory=lambda: embedder)
    if moment is not None:
        with store.lock:
            if store.fts:
                store.db.execute(
                    "DELETE FROM dictations_fts WHERE rowid IN (SELECT id FROM dictations WHERE created_at >= ?)", (moment,)
                )
            store.db.execute("DELETE FROM dictations WHERE created_at >= ?", (moment,))
            store._delete_corrections("created_at >= ?", (moment,))
            store.db.commit()
    store.embedder = embedder
    return store


def read_wav(path: Path) -> Any:
    """Mono float32 samples of a 16 kHz WAV: the archive's own format, and `say`'s here."""
    import numpy as np

    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path.name}: WAV değil")
    offset, fmt, body = 12, None, None
    while offset + 8 <= len(data):
        chunk = data[offset:offset + 4]
        size = int.from_bytes(data[offset + 4:offset + 8], "little")
        content = data[offset + 8:offset + 8 + size]
        if chunk == b"fmt ":
            tag = int.from_bytes(content[0:2], "little")
            if tag == 0xFFFE and len(content) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the real tag is in the sub-format
                tag = int.from_bytes(content[24:26], "little")
            fmt = (tag, int.from_bytes(content[2:4], "little"), int.from_bytes(content[4:8], "little"),
                   int.from_bytes(content[14:16], "little"))
        elif chunk == b"data":
            body = content
        offset += 8 + size + (size & 1)
    if fmt is None or body is None:
        raise ValueError(f"{path.name}: fmt ya da data bölümü yok")
    tag, channels, rate, bits = fmt
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path.name}: {rate} Hz, {SAMPLE_RATE} bekleniyordu")
    if tag == 3 and bits == 32:
        samples = np.frombuffer(body, dtype="<f4")
    elif tag == 1 and bits == 16:
        samples = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"{path.name}: desteklenmeyen biçim (etiket {tag}, {bits} bit)")
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32)


def guard_cases() -> list[dict[str, Any]]:
    """The guard sentences, each with its audio spoken by macOS's Turkish voice."""
    GUARD_AUDIO.mkdir(parents=True, exist_ok=True)
    cases = []
    for item in json.loads(GUARD.read_text(encoding="utf-8")):
        text = item["text"]
        path = GUARD_AUDIO / f"guard-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}.wav"
        if not path.exists():
            subprocess.run(
                ["say", "-v", "Yelda", "--file-format=WAVE", f"--data-format=LEF32@{SAMPLE_RATE}", "-o", str(path), text],
                check=True,
            )
        cases.append({"name": f"koruma/{item['kind']}: {text[:40]}", "audio": path, "expected": text, "moment": None})
    return cases


def keep_recordings(verified: list[dict[str, Any]], archive: Path | None = None, kept: Path | None = None) -> list[str]:
    """Copy the verified recordings still in the archive into the kept folder;
    returns the names copied now. A recording already kept is left alone."""
    archive, kept = archive or ARCHIVE, kept or KEPT
    copied = []
    for case in verified:
        name = case.get("recording")
        if not name or Path(name).name != name:
            continue  # a bare file name only; never a path out of the folders
        source, target = archive / name, kept / name
        if source.is_file() and not target.exists():
            kept.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def recording_cases(
    verified: list[dict[str, Any]], folders: tuple[Path, ...] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """The verified cases whose recording can still be found — in the kept
    folder first, then in the archive — and the names of those that cannot."""
    folders = folders or (KEPT, ARCHIVE)
    cases, missing = [], []
    for case in verified:
        name = case.get("recording")
        if not name:
            continue
        bare = Path(name).name == name  # a file name, never a path out of the folders
        audio = next((folder / name for folder in folders if bare and (folder / name).is_file()), None)
        if audio is None:
            missing.append(name)
            continue
        cases.append({"name": name, "audio": audio, "expected": case["expected"], "moment": recorded_at(name)})
    return cases, missing


# MARK: one case through the chain

def hint_effects(text: str, alternates: list[dict[str, str]], expected: str) -> dict[str, int]:
    """How many second-listening hints would fix a word, break one, or change nothing."""
    before = benchmark.word_errors(expected, text)
    effects = {"doğru": 0, "yanlış": 0, "nötr": 0}
    for pair in alternates:
        after = benchmark.word_errors(expected, benchmark.apply(text, [pair]))
        effects["doğru" if after < before else "yanlış" if after > before else "nötr"] += 1
    return effects


def run_case(case: dict[str, Any], settings: dict[str, Any], chain: dict[str, Any], folder: Path) -> dict[str, Any]:
    worker, np, torch, processor, runtime = chain["worker"], chain["np"], chain["torch"], chain["processor"], chain["runtime"]
    learned = learned_before(settings, case["moment"])
    terms = settings["terms"]
    late_terms = learned_terms(terms, learned)
    store = memory_as_of(case["moment"], folder, chain["embedder"])
    runtime["memory"] = store

    raw_path = folder / "audio.f32"
    read_wav(case["audio"]).astype("<f4").tofile(raw_path)
    request = {
        "audio_path": str(raw_path),
        "language": settings["language"],
        "contexts": settings["contexts"],
        "prompt_terms": terms,
        "learned_terms": late_terms,
        "bias_terms": terms + late_terms,
        "builtin_vocabulary": settings["builtin"],
    }
    started = time.time()
    first, alternates = worker.transcribe(request, np, torch, processor, runtime)
    whisper_seconds = time.time() - started

    corrected, applied = first, []
    smart = chain.get("smart")
    if smart is not None:
        all_terms = terms + late_terms
        vocabulary = all_terms + [term for term in runtime["vocabulary"] if term not in set(all_terms)]
        examples = [{"text": c["context"], "wrong": c["wrong"], "right": c["right"]} for c in learned[:SWIFT_EXAMPLES]]
        seen = {(e["wrong"], e["right"]) for e in examples}
        examples += [e for e in store.nearest_corrections(first) if (e["wrong"], e["right"]) not in seen]
        applied, _ = smart.suggest(first, vocabulary, examples, store.learned_pairs(), alternates)
        corrected = benchmark.apply(first, applied)
    store.db.close()

    expected = case["expected"]
    return {
        "name": case["name"],
        "expected": expected,
        "words": len(benchmark.words(expected)),
        "first": first,
        "alternates": alternates,
        "applied": applied,
        "corrected": corrected,
        "first_errors": benchmark.word_errors(expected, first),
        "corrected_errors": benchmark.word_errors(expected, corrected),
        "hints": hint_effects(first, alternates, expected),
        "whisper_seconds": whisper_seconds,
    }


def load_chain(with_smart: bool) -> dict[str, Any]:
    import memory as memory_store
    import term_llm
    import whisper_worker

    whisper_worker.emit = lambda message: None  # the worker's stdout protocol is not wanted here
    np, torch, processor, _device, _snapshot, runtime = whisper_worker.load_runtime()
    chain = {"worker": whisper_worker, "np": np, "torch": torch, "processor": processor, "runtime": runtime,
             "embedder": memory_store.make_embedder()}
    if with_smart:
        smart = term_llm.SmartCorrector(emit=lambda message: None)
        smart.start()
        if smart.thread is not None:
            smart.thread.join()
        if smart.model is None:
            raise RuntimeError(f"Akıllı düzeltme modeli yüklenemedi: {smart.detail}")
        chain["smart"] = smart
    return chain


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    words = sum(r["words"] for r in rows)
    first = sum(r["first_errors"] for r in rows)
    corrected = sum(r["corrected_errors"] for r in rows)
    hints = {key: sum(r["hints"][key] for r in rows) for key in ("doğru", "yanlış", "nötr")}
    return {
        "cases": len(rows), "words": words,
        "first_errors": first, "first_wer": 100 * first / max(1, words),
        "corrected_errors": corrected, "corrected_wer": 100 * corrected / max(1, words),
        "hints": hints, "whisper_seconds": sum(r["whisper_seconds"] for r in rows) / max(1, len(rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verified", default=str(benchmark.VERIFIED))
    parser.add_argument("--no-smart", action="store_true", help="Whisper only, no smart correction")
    parser.add_argument("--no-guard", action="store_true", help="skip the synthesized guard sentences")
    parser.add_argument("--save", help="write every transcript and the totals to this JSON file")
    args = parser.parse_args()

    verified = benchmark.load_cases(Path(args.verified)) if Path(args.verified).exists() else []
    kept = keep_recordings(verified)
    if kept:
        print(f"{len(kept)} doğrulanmış kayıt arşivden {KEPT} klasörüne kopyalandı.")
    recordings, missing = recording_cases(verified)
    if missing:
        print(f"{len(missing)} doğrulanmış kaydın sesi artık yok (arşivden silinmiş): " + ", ".join(missing))
    sets = {"kayıtlar": recordings}
    if not args.no_guard:
        sets["koruma"] = guard_cases()
    if not sets["kayıtlar"]:
        print(f"Sesi bulunan doğrulanmış kayıt yok ({args.verified}; {KEPT}, {ARCHIVE}).")
        return 1

    settings = read_settings()
    chain = load_chain(with_smart=not args.no_smart)
    results: dict[str, Any] = {"sets": {}}
    with tempfile.TemporaryDirectory(prefix="whisperhold-bench-") as scratch:
        for label, cases in sets.items():
            rows = [run_case(case, settings, chain, Path(scratch)) for case in cases]
            results["sets"][label] = {"rows": rows, "summary": summary(rows)}

    for label, data in results["sets"].items():
        s = data["summary"]
        print(f"\n## {label}: {s['cases']} kayıt, {s['words']} kelime")
        print(f"ilk geçiş: {s['first_errors']} hata (%{s['first_wer']:.1f}) · düzeltme sonrası: "
              f"{s['corrected_errors']} hata (%{s['corrected_wer']:.1f}) · ipucu doğru/yanlış/nötr: "
              f"{s['hints']['doğru']}/{s['hints']['yanlış']}/{s['hints']['nötr']} · Whisper {s['whisper_seconds']:.1f} s/kayıt")
        for r in data["rows"]:
            if r["first_errors"] or r["corrected_errors"] or r["alternates"]:
                print(f"  [{r['first_errors']}→{r['corrected_errors']}] {r['name']}")
                print(f"      ilk:      {r['first']}")
                if r["corrected"] != r["first"]:
                    print(f"      düzeltme: {r['corrected']}")
                if r["alternates"]:
                    print(f"      ipucu:    {r['alternates']}")
    if args.save:
        Path(args.save).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
