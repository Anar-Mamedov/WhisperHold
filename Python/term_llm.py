"""Optional second opinion from a local language model ("akıllı düzeltme").

Whisper hears sounds; it cannot know that a developer who says "bekleyen
arkadaşım" about a colleague meant "backendci arkadaşım". A small instruct
model reading the finished sentence can. It is asked for a short list of
misheard-term replacements only — never for a rewrite — and the app applies a
replacement only when the wrong text occurs verbatim and the proposed term
is in the vocabulary (the user's list or the bundled one) or a Turkish-suffixed
form of such a term. That bounds the damage a confused model can do to a
handful of known words.

The model (Qwen3-4B-Instruct-2507, Apache-2.0, ~8 GB fp16) is downloaded on
first use into the app's model cache and loaded next to Whisper on the GPU.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW_TOKENS = 160
MAX_REPLACEMENTS = 8
MAX_SUFFIX_LETTERS = 5

SYSTEM_PROMPT = (
    "You review a Turkish speech-to-text transcript spoken by a software developer. "
    "The recognizer sometimes replaces an English software term with a similar-sounding "
    "Turkish word. Your job is to find ONLY those words and give the intended term, keeping "
    "any Turkish suffix attached. Rules: "
    "(1) Propose a replacement only when the Turkish word makes no sense in its sentence AND "
    "the software term fits exactly where it stands. "
    "(2) Ordinary Turkish words such as komite, bekleyen, model, servis, depo, tablo, "
    "bütün are usually correct; a sentence that reads naturally in Turkish needs no change. "
    "(3) Never change grammar, punctuation, word order or anything else. "
    "(4) When in doubt, change nothing. "
    'Answer with JSON only: [{"wrong": "<exact word from the transcript>", "right": "<term with the same suffix>"}] '
    "or [] when nothing is wrong."
)
FEW_SHOT = [
    (
        "Bu dokümanı ve içeriklerini hepsini bekleyen arkadaşım bana verdi.",
        '[{"wrong": "bekleyen", "right": "backendci"}]',
    ),
    (
        "Projeyi bild ettim, komit attım, endpoynt bozuk.",
        '[{"wrong": "bild", "right": "build"}, {"wrong": "komit", "right": "commit"}, '
        '{"wrong": "endpoynt", "right": "endpoint"}]',
    ),
    ("Bekleyen arkadaşım sonunda geldi, modeli dün eğittik.", "[]"),
    ("Komite toplandı ve servis gecikti, bütün butonlar bozuldu.", "[]"),
    ("Bu konuda doküman ne söylüyor, depodan transfer ettik.", "[]"),
    ("Button'lar bozuldu, tablo kaydı, servis çalışıyor.", "[]"),
    ("Backendci arkadaşım dokümanı verdi, pull request'i merge ettim.", "[]"),
]

# A proposed form may extend a vocabulary term only by Turkish suffixes
# (backend → backendci, pipeline → pipeline'ı), never by English inflection
# (button → buttons).
_TURKISH_SUFFIX = re.compile(
    r"^(?:ci|cu|ler|lar|leri|lari|lerin|larin|i|u|yi|yu|e|a|ye|ya|ne|na|de|da|te|ta|nde|nda|den|dan|ten|tan|"
    r"nden|ndan|in|un|nin|nun|le|la|yle|yla|im|um|iz|uz|miz|muz|niz|nuz|ki|li|lu|si|su|"
    r"dim|dum|din|dun|di|du|dik|duk|tim|tum|tin|tun|ti|tu|tik|tuk|yor|yorum|yorsun|yoruz|mek|mak|me|ma|mis|mus|"
    r"yip|yup|ip|up|sin|sun|lim|lum|ecek|acak|se|sa|ken|erek|arak|ince|inca)+$"
)


def fold(text: str) -> str:
    return text.lower().replace("ı", "i").replace("i̇", "i").replace("'", "").replace("’", "")


def occurs(wrong: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(wrong) + r"(?!\w)", text, re.IGNORECASE) is not None


_SOUND_ALIKE = str.maketrans({"c": "k", "q": "k", "w": "v", "x": "k", "ç": "c", "ş": "s", "ğ": "g", "ö": "o", "ü": "u"})


def sounds_alike(wrong: str, right: str) -> bool:
    """A misheard term starts with the same sound and has a similar length.

    This is what separates an acoustic confusion ("bekleyen" for "backendci",
    "komite" for "commit") from a rewrite the model simply felt like making
    ("grep" → "deploy"). It keeps the language model in its lane.
    """
    a = fold(wrong).translate(_SOUND_ALIKE)
    b = fold(right).translate(_SOUND_ALIKE)
    if not a or not b or a[0] != b[0]:
        return False
    longer, shorter = max(len(a), len(b)), min(len(a), len(b))
    return longer <= 2 * shorter + 2


def derived_from(word: str, folded_vocabulary: set[str]) -> bool:
    """Is the folded word a vocabulary term, or one with a Turkish suffix glued on?"""
    if word in folded_vocabulary:
        return True
    return any(
        len(term) >= 3
        and word.startswith(term)
        and 0 < len(word) - len(term) <= MAX_SUFFIX_LETTERS
        and _TURKISH_SUFFIX.match(word[len(term):]) is not None
        for term in folded_vocabulary
    )


def validate(pairs: list[Any], text: str, vocabulary: list[str]) -> list[dict[str, str]]:
    """Keep proposals that are safe to apply: verbatim source that is not already a
    term, similar sound, vocabulary-derived target."""
    folded_vocabulary = {fold(term) for term in vocabulary if term.strip()}
    kept: list[dict[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        wrong = str(pair.get("wrong", "")).strip()
        right = str(pair.get("right", "")).strip()
        if not wrong or not right or fold(wrong) == fold(right):
            continue
        if not occurs(wrong, text) or not sounds_alike(wrong, right):
            continue
        # A word that already is a term (or a term with its suffix, "backendci")
        # is not misheard; the model once proposed trimming it to "backend".
        if derived_from(fold(wrong), folded_vocabulary):
            continue
        if not derived_from(fold(right), folded_vocabulary):
            continue
        kept.append({"wrong": wrong, "right": right})
        if len(kept) >= MAX_REPLACEMENTS:
            break
    return kept


def parse_answer(answer: str) -> list[Any]:
    match = re.search(r"\[.*\]", answer, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


MAX_EXAMPLES = 10


def build_messages(text: str, examples: list[dict[str, str]]) -> list[dict[str, str]]:
    """System prompt, the fixed examples, then the user's own past corrections.

    A correction the user made by hand ("in this sentence 'bekleyen' meant
    'backendci'") is the best evidence of how they speak; it goes in as one
    more worked example, context included, so the model applies it only in a
    sentence like that one and not to every "bekleyen".
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for question, answer in FEW_SHOT:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})
    for example in examples[:MAX_EXAMPLES]:
        context = str(example.get("text", "")).strip()
        wrong = str(example.get("wrong", "")).strip()
        right = str(example.get("right", "")).strip()
        if not context or not wrong or not right or not occurs(wrong, context):
            continue
        messages.append({"role": "user", "content": context})
        messages.append({"role": "assistant", "content": json.dumps([{"wrong": wrong, "right": right}], ensure_ascii=False)})
    messages.append({"role": "user", "content": text})
    return messages


class SmartCorrector:
    """Loads the model in a background thread; `suggest` only works once ready."""

    def __init__(self, emit: Any) -> None:
        self.emit = emit
        self.state = "off"
        self.detail = ""
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def report(self, state: str, detail: str = "") -> None:
        self.state, self.detail = state, detail
        self.emit({"event": "llm_state", "state": state, "detail": detail})

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        if self.state == "ready":
            self.report("ready", self.detail)
            return
        self.thread = threading.Thread(target=self._load, name="smart-correction-load", daemon=True)
        self.thread.start()

    def _load(self) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer

            patterns = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*", "*.jinja"]
            try:
                snapshot = snapshot_download(MODEL_ID, allow_patterns=patterns, local_files_only=True)
                self.report("loading", "Akıllı düzeltme modeli yükleniyor…")
            except Exception:
                self.report("downloading", "Akıllı düzeltme modeli indiriliyor (~8 GB, bir kez)…")
                snapshot = snapshot_download(MODEL_ID, allow_patterns=patterns)
                self.report("loading", "Akıllı düzeltme modeli yükleniyor…")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, dtype=dtype)
            model.eval()
            # Weights go to the GPU only while no dictation is being decoded.
            with gpu_lock.LOCK:
                model.to(device)
            with self.lock:
                self.tokenizer, self.model, self.device = tokenizer, model, device
            self.report("ready", f"{MODEL_ID} · {device.upper()}")
        except Exception as error:  # noqa: BLE001 - reported to the app, never fatal
            self.report("failed", f"Akıllı düzeltme modeli yüklenemedi: {error}")

    def wait_if_loading(self, seconds: float) -> None:
        """A load from disk takes seconds; a download takes minutes. Only wait for the former."""
        if self.state == "loading" and self.thread is not None:
            self.thread.join(timeout=seconds)

    def suggest(
        self, text: str, vocabulary: list[str], examples: list[dict[str, str]] | None = None
    ) -> tuple[list[dict[str, str]], str]:
        self.wait_if_loading(20.0)
        with self.lock:
            model, tokenizer = self.model, self.tokenizer
        if model is None or not text.strip():
            return [], self.state
        import torch

        messages = build_messages(text, examples or [])
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False
        )
        started = time.time()
        with gpu_lock.LOCK, torch.inference_mode():
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        answer = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        proposals = parse_answer(answer)
        kept = validate(proposals, text, vocabulary)
        print(
            f"akıllı düzeltme {time.time() - started:.1f} s: önerilen {proposals!r} → uygulanan {kept!r}",
            file=sys.stderr,
            flush=True,
        )
        return kept, "ready"
