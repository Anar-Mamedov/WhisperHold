#!/usr/bin/env python3
"""Measure the correction layers on the user's own dictations.

The memory already holds most of what a benchmark needs: every correction
the user taught (the sentence as Whisper wrote it and the intended word) and
every dictation they left alone. This tool turns them into a test set and
runs the worker's text pipeline over it — spelled-out acronyms, the
smart-correction model with the memory's examples and the second listening's
alternates, the gate — reporting word errors fixed against word errors
introduced. Run it after changing a model, a prompt or the gate; the numbers
settle the argument.

A dictation left alone is only presumed right: the user may not have noticed
an error, and listening to the recordings found dictations stored with a
non-word in them. Cases checked against the recording go in a verified set,
`benchmark_verified.json` next to the memory, and replace whatever the memory
presumed about the same dictation. It is the user's own speech, so it lives
with their data and never in this repository.

    cd Python
    ~/Library/Application\\ Support/WhisperHold/runtime/.venv/bin/python3 benchmark.py
    ... --set my_cases.json      # a hand-written set instead of the memory
    ... --verified other.json    # another verified set (default: next to the memory)
    ... --no-model               # the deterministic layers only
    ... --save results.json      # keep the proposals for a later re-score

A set file (and the verified set) is a JSON list of {"raw": ..., "expected":
...} objects; "expected" equal to "raw" marks a sentence that must not
change. "raw" is the first pass as the worker returned it; an optional
"alternates" list ([{"wrong": ..., "right": ...}], what the second listening
heard differently) reaches the model and the gate as it does in the app.
Leave-one-out: when a sentence is scored, the corrections the user taught on
that very dictation are left out, so the model has to generalise, as it
would have had to the first time.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/Library/Application Support/WhisperHold/HuggingFace"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import acronyms  # noqa: E402
import memory as memory_store  # noqa: E402
import term_bias  # noqa: E402
import term_llm  # noqa: E402

SUPPORT = Path.home() / "Library/Application Support/WhisperHold"
VERIFIED = SUPPORT / "benchmark_verified.json"


def words(text: str) -> list[str]:
    text = text.lower().replace("i̇", "i")
    return re.sub(r"[^\w\s']", " ", text).split()


def word_errors(reference: str, hypothesis: str) -> int:
    ref, hyp = words(reference), words(hypothesis)
    errors = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes():
        if tag != "equal":
            errors += max(i2 - i1, j2 - j1)
    return errors


def apply(text: str, pairs: list[dict[str, str]]) -> str:
    for pair in pairs:
        text = re.sub(r"(?<!\w)" + re.escape(pair["wrong"]) + r"(?!\w)", lambda _m: pair["right"], text, flags=re.IGNORECASE)
    return text


def set_from_memory(store: memory_store.Memory) -> list[dict[str, str]]:
    """Positives from the taught corrections, negatives from untouched dictations."""
    items = []
    seen = set()
    with store.lock:
        corrections = store.db.execute("SELECT wrong, right, context FROM corrections ORDER BY id").fetchall()
        dictations = store.db.execute("SELECT text FROM dictations ORDER BY id").fetchall()
    for row in corrections:
        raw = row["context"].strip()
        if not raw or not term_llm.occurs(row["wrong"], raw):
            continue
        expected = apply(raw, [{"wrong": row["wrong"], "right": row["right"]}])
        if expected != raw and raw not in seen:
            seen.add(raw)
            items.append({"raw": raw, "expected": expected, "pair": [row["wrong"], row["right"]]})
    positives = [it["raw"] for it in items]
    for row in dictations:
        text = row["text"].strip()
        if len(text.split()) < 3 or text in seen or any(p in text or text in p for p in positives):
            continue
        seen.add(text)
        items.append({"raw": text, "expected": text})
    return items


def load_cases(path: Path) -> list[dict]:
    """A set file: {raw, expected} objects, optionally with the second listening's alternates."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"{path}: bir JSON listesi bekleniyordu")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("raw"), str) or not isinstance(case.get("expected"), str):
            raise ValueError(f"{path}: her öğede raw ve expected metni olmalı: {case!r}")
        alternates = case.get("alternates", [])
        if not isinstance(alternates, list) or not all(isinstance(a, dict) for a in alternates):
            raise ValueError(f"{path}: alternates bir {{wrong, right}} listesi olmalı: {case!r}")
    return cases


def same_dictation(a: str, b: str) -> bool:
    """Whether two texts are one dictation. The memory keeps the text after
    correction and a verified case the first pass, so they may differ in a word or two."""
    length = max(len(words(a)), len(words(b)))
    return length > 0 and word_errors(a, b) <= max(2, length // 4)


def with_verified(items: list[dict], verified: list[dict]) -> list[dict]:
    """The verified cases, then the memory's cases about every other dictation."""
    def covered(text: str) -> bool:
        return any(same_dictation(text, case["raw"]) or same_dictation(text, case["expected"]) for case in verified)

    return list(verified) + [it for it in items if not covered(it["raw"])]


def taught_on(raw: str, corrections: list[dict]) -> set[str]:
    """Folded sources of the corrections the user taught on this very dictation:
    a correction's context is the sentence it was made in, so it occurs in the raw text."""
    folded = term_llm.fold(raw)
    return {
        term_llm.fold(c["wrong"]) for c in corrections
        if c["context"].strip() and term_llm.fold(c["context"].strip()) in folded
    }


def run_model(
    it: dict, text: str, smart: object, store: object, corrections: list[dict],
    learned_all: list[tuple[str, str]], vocabulary: list[str],
) -> tuple[list, list[dict[str, str]]]:
    """The worker's suggest step for one case: the memory's nearest examples and
    taught pairs (leave-one-out), the second listening's alternates, the model, the gate."""
    own = taught_on(it["raw"], corrections)
    examples = [
        e for e in store.nearest_corrections(text)
        if e["text"].strip() != it["raw"].strip() and term_llm.fold(e["wrong"]) not in own
    ]
    learned = [p for p in learned_all if term_llm.fold(p[0]) not in own]
    alternates = [a for a in it.get("alternates") or [] if isinstance(a, dict)]
    messages = term_llm.build_messages(text, examples, alternates)
    proposals = term_llm.merge_proposals([term_llm.parse_answer(a) for a in smart.answers(messages)])
    taught = learned + [(e["wrong"], e["right"]) for e in examples]
    return proposals, term_llm.validate(proposals, text, vocabulary, taught, alternates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", help="JSON list of {raw, expected} instead of the memory's own cases")
    parser.add_argument("--verified", default=str(VERIFIED),
                        help="cases checked against the recording; they replace the memory's guesses (ignored with --set)")
    parser.add_argument("--no-model", action="store_true", help="deterministic layers only")
    parser.add_argument("--save", help="write proposals and scores to this JSON file")
    parser.add_argument("--memory", default=str(SUPPORT / "memory.sqlite"))
    args = parser.parse_args()

    store = memory_store.Memory(Path(args.memory), embedder_factory=memory_store.make_embedder)
    if args.set:
        items = load_cases(Path(args.set))
    else:
        items = set_from_memory(store)
        if Path(args.verified).exists():
            verified = load_cases(Path(args.verified))
            items = with_verified(items, verified)
            print(f"{len(verified)} doğrulanmış kayıt ({args.verified})", flush=True)
    if not items:
        print("Ölçülecek bir şey yok: hafızada düzeltme ya da dikte bulunamadı.")
        return 1
    learned_all = store.learned_pairs()
    with store.lock:
        corrections = [dict(row) for row in store.db.execute("SELECT wrong, right, context FROM corrections")]
    vocabulary = list(dict.fromkeys([right for _, right in learned_all] + term_bias.load_vocabulary()))
    print(f"{sum(1 for it in items if it['raw'] != it['expected'])} hatalı, "
          f"{sum(1 for it in items if it['raw'] == it['expected'])} değişmemesi gereken cümle", flush=True)

    smart = None
    if not args.no_model:
        store.start_embedder()
        store.thread.join() if store.thread else None
        smart = term_llm.SmartCorrector(emit=lambda message: None)
        smart.start()
        smart.thread.join()
        if smart.model is None:
            print(f"Model yüklenemedi: {smart.detail}")
            return 1
        print(f"model: {smart.detail}", flush=True)

    results = []
    started = time.time()
    for it in items:
        text = acronyms.restore(it["raw"], vocabulary)
        applied: list[dict[str, str]] = []
        proposals: list = []
        if smart is not None:
            proposals, applied = run_model(it, text, smart, store, corrections, learned_all, vocabulary)
            text = apply(text, applied)
        before, after = word_errors(it["expected"], it["raw"]), word_errors(it["expected"], text)
        results.append({**it, "output": text, "proposals": proposals, "applied": applied, "before": before, "after": after})

    positives = [r for r in results if r["raw"] != r["expected"]]
    negatives = [r for r in results if r["raw"] == r["expected"]]
    fixed = sum(1 for r in positives if r["before"] and r["after"] == 0)
    damaged = [r for r in negatives if r["after"] > 0]
    print(f"\nhatalı cümleler: {sum(r['before'] for r in positives)} → {sum(r['after'] for r in positives)} kelime hatası, "
          f"{fixed}/{len(positives)} cümle tamamen düzeldi")
    print(f"doğru cümleler: {len(damaged)}/{len(negatives)} bozuldu")
    if smart is not None:
        print(f"süre: {(time.time() - started) / max(1, len(items)):.1f} s / cümle")
    for r in positives:
        mark = "✓" if r["before"] and r["after"] == 0 else ("~" if r["after"] < r["before"] else "✗")
        print(f"  {mark} {r['raw'][:70]:70s} → {json.dumps(r['applied'], ensure_ascii=False)[:60]}")
    for r in damaged:
        print(f"  BOZULDU {r['raw'][:70]:70s} → {json.dumps(r['applied'], ensure_ascii=False)[:60]}")
    if args.save:
        Path(args.save).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
