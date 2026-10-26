#!/usr/bin/env python3
"""Measure the correction layers on the user's own dictations.

The memory already holds everything a benchmark needs: every correction the
user taught (the sentence as Whisper wrote it and the intended word) and
every dictation they left alone. This tool turns them into a test set and
runs the worker's text pipeline over it — spelled-out acronyms, the
smart-correction model with the memory's examples, the gate — reporting
word errors fixed against word errors introduced. Run it after changing a
model, a prompt or the gate; the numbers settle the argument.

    cd Python
    ~/Library/Application\\ Support/WhisperHold/runtime/.venv/bin/python3 benchmark.py
    ... --set my_cases.json      # a hand-written set instead of the memory
    ... --no-model               # the deterministic layers only
    ... --save results.json      # keep the proposals for a later re-score

A set file is a JSON list of {"raw": ..., "expected": ...} objects; "expected"
equal to "raw" marks a sentence that must not change. Leave-one-out: when a
sentence is scored, the memory example made from that very sentence is left
out, so the model has to generalise, as it would have had to the first time.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", help="JSON list of {raw, expected} instead of the memory's own cases")
    parser.add_argument("--no-model", action="store_true", help="deterministic layers only")
    parser.add_argument("--save", help="write proposals and scores to this JSON file")
    parser.add_argument("--memory", default=str(SUPPORT / "memory.sqlite"))
    args = parser.parse_args()

    store = memory_store.Memory(Path(args.memory), embedder_factory=memory_store.make_embedder)
    items = json.loads(Path(args.set).read_text(encoding="utf-8")) if args.set else set_from_memory(store)
    if not items:
        print("Ölçülecek bir şey yok: hafızada düzeltme ya da dikte bulunamadı.")
        return 1
    learned_all = store.learned_pairs()
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
            examples = [e for e in store.nearest_corrections(text) if e["text"].strip() != it["raw"].strip()]
            pair = it.get("pair")
            if pair:  # leave-one-out: the pair taught from this very sentence
                examples = [e for e in examples if term_llm.fold(e["wrong"]) != term_llm.fold(pair[0])]
            learned = [p for p in learned_all if not pair or term_llm.fold(p[0]) != term_llm.fold(pair[0])]
            messages = term_llm.build_messages(text, examples)
            proposals = term_llm.merge_proposals([term_llm.parse_answer(a) for a in smart.answers(messages)])
            applied = term_llm.validate(proposals, text, vocabulary, learned + [(e["wrong"], e["right"]) for e in examples])
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
