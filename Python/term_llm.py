"""Optional second opinion from a local language model ("akıllı düzeltme").

Whisper hears sounds; it cannot know that a developer who says "bekleyen
arkadaşım" about a colleague meant "backendci arkadaşım", or that "çalışırsan"
in "kayıt almaya çalışırsan böyle bir sorunla karşılaştım" was "çalışırken".
A local instruct model reading the finished sentence can. It is asked for a
short list of word replacements only — never for a rewrite — and the worker
applies a replacement only when it passes the gate in `validate`: the word
occurs verbatim and is not already a term, and the replacement sounds like it
(same first sound, at most a third of the letters changed) and is a word the
macOS dictionary or the vocabulary knows — or the user taught exactly that
pair. Measured on the user's own transcripts (24 hearing errors, 51 correct
dictations) this fixed what the model found and changed no correct sentence.

The model is Gemma 4 12B (QAT 4-bit, ~11 GB) through Apple's MLX when
mlx-lm is installed; otherwise Qwen3-4B-Instruct-2507 (~8 GB fp16) through
PyTorch. It is downloaded on first use into the app's model cache and loaded
next to Whisper on the GPU, behind the same lock. Several MLX models can be
listed (WHISPERHOLD_LLM); their proposals are then merged.
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
import dictionary  # noqa: E402
import gpu_lock  # noqa: E402

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# The MLX model, used when mlx-lm is installed (see requirements-optional.txt),
# chosen by measurement on the user's own dictation errors (README, "Smart
# correction"). Gemma 4 12B fixed the real-word mistakes the user makes most
# ("çıktığının" → "çıktının", "çalışırsan" → "çalışırken") and, behind the
# gate, changed no correct sentence. The 26B mixture of experts caught a
# different few and the two combined caught the most, but holding both next
# to Whisper took about 30 GB; the user chose the 12B alone for that reason.
# WHISPERHOLD_LLM overrides the list (comma-separated model ids): with two or
# more models the proposals are merged, the first model winning on a word
# they disagree on.
MLX_MODELS = ["mlx-community/gemma-4-12B-it-qat-4bit"]
# The bigger model for an explicit "look again" from the menu: loaded for that
# one request and dropped afterwards, so it costs memory only while it runs.
# Measured next to the 12B on the user's errors: it fixes a different few
# ("RVD" → "RWD", "yaptınca" → "yaptın ki"), never touched a correct sentence.
DEEP_MODEL_ID = os.environ.get("WHISPERHOLD_DEEP_LLM") or "mlx-community/gemma-4-26b-a4b-it-4bit"
MLX_MODEL_SIZES = {"mlx-community/gemma-4-12B-it-qat-4bit": "~11 GB", "mlx-community/gemma-4-26b-a4b-it-4bit": "~15 GB"}



def mlx_model_ids() -> list[str]:
    override = os.environ.get("WHISPERHOLD_LLM", "").strip()
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    return list(MLX_MODELS)


def merge_proposals(answers: list[list[Any]]) -> list[dict[str, Any]]:
    """Proposals of several models, the first model's answer winning for a
    word they disagree on (measured: the 26B's "yaptın ki" over the 12B's
    "yapınca")."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, proposals in enumerate(answers):
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            key = fold(str(proposal.get("wrong", "")))
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append({**proposal, "model": index})
    return merged
MAX_NEW_TOKENS = 160
STUCK_LOAD_SECONDS = 120
MAX_REPLACEMENTS = 8
MAX_SUFFIX_LETTERS = 5

SYSTEM_PROMPT = (
    "You review a Turkish speech-to-text transcript spoken by a software developer. "
    "The recognizer sometimes writes a similar-sounding but wrong word: an English software term as a Turkish word "
    "(\"bekleyen\" for \"backendci\"), a non-word (\"müsayit\" for \"müsait\"), spelled-out letter names for an acronym "
    "(\"ey pi ay\" for \"API\"), or a real Turkish word that does not fit its sentence (\"hazırlarsan\" for \"hazırlarken\"). "
    "Your job is to find ONLY such words and give the intended word, keeping any suffix. Rules: "
    "(1) Propose a replacement only when the written word does not fit its sentence AND a similar-sounding word fits exactly where it stands. "
    "(2) The replacement must sound like the written word: same first sound, similar length. Never replace a word with a synonym or a better wording. "
    "(3) A sentence that reads naturally in Turkish needs no change. Never change grammar, style, word order, punctuation or capitalization. "
    "(4) When in doubt, change nothing. "
    'Answer with JSON only: [{"wrong": "<exact word(s) from the transcript>", "right": "<intended word(s) with the same suffix>"}] or [] when nothing is wrong.'
)
FEW_SHOT = [
    ("Toplantıyı yarına erteleyelim, herkes müsayit olsun.", '[{"wrong": "müsayit", "right": "müsait"}]'),
    ("Raporu hazırlarsan dün gece bitirdim, sabah gönderirim.", '[{"wrong": "hazırlarsan", "right": "hazırlarken"}]'),
    ("Bunu ey pi ay üzerinden çağırıyoruz, cevap JSON dönüyor.", '[{"wrong": "ey pi ay", "right": "API"}]'),
    ("Bu dokümanı ve içeriklerini hepsini bekleyen arkadaşım bana verdi.", '[{"wrong": "bekleyen", "right": "backendci"}]'),
    (
        "Projeyi bild ettim, komit attım, endpoynt bozuk.",
        '[{"wrong": "bild", "right": "build"}, {"wrong": "komit", "right": "commit"}, {"wrong": "endpoynt", "right": "endpoint"}]',
    ),
    ("Kodu yazarsan test etmeyi unutma, bekleyen işler listede duruyor.", "[]"),
    ("Komite toplandı ve servis gecikti, bütün butonlar bozuldu.", "[]"),
    ("Dün akşam arabayla eve dönerken yolda çok trafik vardı.", "[]"),
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


# How a word sounds to a Turkish ear, roughly one letter per sound, so that
# "backendci" and "be kentçi" come out two edits apart ("bakendci" / "bekentci").
_DIGRAPHS = [("tch", "c"), ("ck", "k"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("th", "t"), ("x", "ks"), ("w", "v"), ("q", "k")]
_LETTERS = str.maketrans({"ç": "c", "ş": "s", "ğ": "", "ö": "o", "ü": "u", "â": "a", "î": "i", "û": "u"})
MAX_EDITS = 3
EDIT_SHARE = 0.34
SENTENCE_PUNCTUATION = re.compile(r"[.?!,;:]")


_HARD_C = re.compile(r"c(?=[aouâ]|[bdfgklmnprstvz]|$)")


def sound(text: str) -> str:
    folded = fold(text).replace(" ", "").replace("-", "")
    for digraph, letter in _DIGRAPHS:
        folded = folded.replace(digraph, letter)
    # English "c" before a, o, u or a consonant is a k ("commit" → "kommit");
    # before e, i, y it stays, which is also the Turkish "c" of "backendci".
    return _HARD_C.sub("k", folded).translate(_LETTERS)


def stem(text: str) -> str:
    """The part before an apostrophe suffix: "pipeline'ı" → "pipeline"."""
    return re.split(r"['’]", text, maxsplit=1)[0]


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def sounds_alike(wrong: str, right: str) -> bool:
    """A hearing error keeps the first sound and changes at most a third of the
    letters (never more than three) — the budget the term corrector was tuned
    to. "be kentçi" → "backendci" (2 of 8) and "RVD" → "RWD" (0) pass;
    "Giral" → "Git" (3 of 5), "komite" → "commit" (3 of 6) and "e-postada" →
    "endpoint'te" do not. Those three were real proposals of a 12B model that
    a looser rule had let through.
    """
    # A suffix only one side carries ("paypline" → "pipeline'ı") is not a difference in sound.
    if ("'" in right or "’" in right) and not ("'" in wrong or "’" in wrong):
        right = stem(right)
    elif ("'" in wrong or "’" in wrong) and not ("'" in right or "’" in right):
        wrong = stem(wrong)
    a, b = sound(wrong), sound(right)
    if not a or not b or a[0] != b[0]:
        return False
    distance = edit_distance(a, b)
    return distance <= MAX_EDITS and distance <= round(EDIT_SHARE * max(len(a), len(b)))


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


def known_word(word: str, folded_vocabulary: set[str]) -> bool:
    """A vocabulary term (with a Turkish suffix at most) or a word the macOS
    dictionary knows. Without the dictionary, terms only."""
    if derived_from(fold(word), folded_vocabulary):
        return True
    return dictionary.is_word(word) is True


def turkish_phrase(text: str) -> bool:
    """Every word is in the Turkish dictionary ("komite toplantısı" yes,
    "be kentçi" no). False when the dictionary is unavailable."""
    words = [word for word in text.split() if any(ch.isalpha() for ch in word)]
    return bool(words) and all(dictionary.is_turkish_word(word) is True for word in words)


def validate(
    pairs: list[Any],
    text: str,
    vocabulary: list[str],
    learned: list[tuple[str, str]] | None = None,
    alternates: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Keep the proposals that are safe to apply.

    Always: the source occurs verbatim and is not already a term. Then either
    the user taught exactly this pair (a learned correction: no further
    questions), or the replacement sounds like the source, carries no
    sentence punctuation, and every word of it is a dictionary word or a
    vocabulary term, and a single real Turkish word is never turned into a
    term. Measured on the user's own transcripts (24 hearing errors, 51 correct
    dictations) and on a synthetic set in their register: the gate let the
    Gemma 4 models fix the errors they found and changed no correct sentence.
    """
    folded_vocabulary = {fold(term) for term in vocabulary if term.strip()}
    taught = {(fold(wrong), fold(right)) for wrong, right in (learned or []) if wrong.strip() and right.strip()}
    # What the second decoding heard is acoustic evidence: such a pair needs no
    # sound check, only the dictionary one.
    heard = {(fold(str(a.get("wrong", ""))), fold(str(a.get("right", "")))) for a in alternates or [] if isinstance(a, dict)}
    kept: list[dict[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        wrong = str(pair.get("wrong", "")).strip()
        right = str(pair.get("right", "")).strip()
        if not wrong or not right or fold(wrong) == fold(right):
            continue
        if not occurs(wrong, text):
            continue
        # A word that already is a term (or a term with its suffix, "backendci")
        # is not misheard; the model once proposed trimming it to "backend".
        if derived_from(fold(wrong), folded_vocabulary):
            continue
        if (fold(wrong), fold(right)) not in taught:
            if SENTENCE_PUNCTUATION.search(right):
                continue
            if (fold(wrong), fold(right)) not in heard and not sounds_alike(wrong, right):
                continue
            if not all(known_word(word, folded_vocabulary) for word in right.split()):
                continue
            # A single real Turkish word never becomes a term on the model's
            # say-so: "Komite toplantısı" once came back as "Commit" (the 12B,
            # on a synthetic test), the kind of proposal the sound budget alone
            # cannot always stop. Such a pair is applied only when taught
            # (above). Real word to real word ("çalışırsan" → "çalışırken")
            # and split terms ("front ent" → "Frontend", whose halves happen
            # to be dictionary words) stay open.
            if len(wrong.split()) == 1 and turkish_phrase(wrong) and not turkish_phrase(right):
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


def with_alternates(text: str, alternates: list[dict[str, str]] | None) -> str:
    """The transcript, followed by what a second decoding of the same audio
    heard differently — a hint, phrased so the model weighs it against the
    sentence rather than applying it blindly."""
    pairs = [(str(a.get("wrong", "")).strip(), str(a.get("right", "")).strip()) for a in alternates or [] if isinstance(a, dict)]
    pairs = [(w, r) for w, r in pairs if w and r and fold(w) != fold(r)]
    if not pairs:
        return text
    heard = "; ".join(f'"{w}" as "{r}"' for w, r in pairs[:MAX_REPLACEMENTS])
    return f"{text}\n\n(A second decoding of the same audio heard {heard}. Use it only where it fits the sentence.)"


def build_messages(
    text: str, examples: list[dict[str, str]], alternates: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
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
    messages.append({"role": "user", "content": with_alternates(text, alternates)})
    return messages


class SmartCorrector:
    """Loads the model in a background thread; `suggest` only works once ready."""

    def __init__(self, emit: Any, model_ids: list[str] | None = None, event: str = "llm_state") -> None:
        self.emit = emit
        self.model_ids = model_ids
        self.event = event
        self.state = "off"
        self.detail = ""
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self.backend = "transformers"
        self.mlx_stream: Any = None
        self.loading_since = 0.0
        # MLX: every loaded (model id, model, tokenizer); `model` above is the first.
        self.mlx_models: list[tuple[str, Any, Any]] = []
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def report(self, state: str, detail: str = "") -> None:
        if state == "loading" and self.state != "loading":
            self.loading_since = time.time()
        self.state, self.detail = state, detail
        self.emit({"event": self.event, "state": state, "detail": detail})

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        if self.state == "ready":
            self.report("ready", self.detail)
            return
        # MLX streams belong to the thread that made them: a model loaded on the
        # loader thread's own default stream cannot be run from the main thread
        # ("There is no Stream(gpu, 1) in current thread"). So the main thread's
        # stream is captured here and the loader works inside it.
        try:
            import mlx.core as mx

            self.mlx_stream = mx.default_stream(mx.default_device())
        except Exception:  # noqa: BLE001 - no MLX: the transformers path
            self.mlx_stream = None
        self.thread = threading.Thread(target=self._load, name="smart-correction-load", daemon=True)
        self.thread.start()

    def _load(self) -> None:
        try:
            try:
                import mlx_lm  # noqa: F401
            except Exception:  # noqa: BLE001 - no MLX on this machine: the transformers model
                self._load_transformers()
            else:
                self._load_mlx()
        except Exception as error:  # noqa: BLE001 - reported to the app, never fatal
            self.report("failed", f"Akıllı düzeltme modeli yüklenemedi: {error}")

    def _snapshot(self, model_id: str, patterns: list[str] | None, size_hint: str) -> str:
        from huggingface_hub import snapshot_download

        try:
            snapshot = snapshot_download(model_id, allow_patterns=patterns, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet
            self.report("downloading", f"Akıllı düzeltme modeli indiriliyor ({size_hint}, bir kez)…")
            snapshot = snapshot_download(model_id, allow_patterns=patterns)
        self.report("loading", "Akıllı düzeltme modeli yükleniyor…")
        return snapshot

    def _load_mlx(self) -> None:
        """The MLX models: Apple's own framework, 4-bit weights, from the model cache.

        Each model is loaded under the GPU lock on its own, so a dictation can
        run between two loads, and becomes usable as soon as it is in."""
        import mlx.core as mx
        from mlx_lm import load

        loaded: list[tuple[str, Any, Any]] = []
        for model_id in self.model_ids or mlx_model_ids():
            snapshot = self._snapshot(model_id, None, MLX_MODEL_SIZES.get(model_id, "birkaç GB"))
            with gpu_lock.LOCK, mx.stream(self.mlx_stream):
                model, tokenizer = load(snapshot)
            loaded.append((model_id, model, tokenizer))
            with self.lock:
                self.mlx_models = list(loaded)
                self.tokenizer, self.model, self.device, self.backend = loaded[0][2], loaded[0][1], "mlx", "mlx"
        names = " + ".join(model_id.split("/")[-1] for model_id, _, _ in loaded)
        self.report("ready", f"{names} · MLX")

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        patterns = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*", "*.jinja"]
        snapshot = self._snapshot(MODEL_ID, patterns, "~8 GB")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, dtype=dtype)
        model.eval()
        # Weights go to the GPU only while no dictation is being decoded.
        with gpu_lock.LOCK:
            model.to(device)
        with self.lock:
            self.tokenizer, self.model, self.device, self.backend = tokenizer, model, device, "transformers"
        self.report("ready", f"{MODEL_ID} · {device.upper()}")

    def unload(self) -> None:
        """Drop the weights and give the memory back (the on-demand deep model)."""
        with self.lock:
            self.model, self.tokenizer, self.mlx_models = None, None, []
        import gc

        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass
        self.report("off", "")

    def wait_if_loading(self, seconds: float) -> None:
        """A load from disk takes seconds; a download takes minutes. Only wait
        for the former — and once at least one MLX model is in, use it. A load
        that has been running for minutes is stuck (it happened once, in a
        library's GPU wait); dictations then stop waiting for it."""
        if self.state == "loading" and self.thread is not None and self.model is None:
            if time.time() - self.loading_since > STUCK_LOAD_SECONDS:
                self.report("failed", "Akıllı düzeltme modeli yüklenemedi: yükleme takıldı")
                return
            self.thread.join(timeout=seconds)

    def answers(self, messages: list[dict[str, str]]) -> list[str]:
        """Every loaded model's reply to the chat, in load order."""
        with self.lock:
            backend, models = self.backend, list(self.mlx_models)
        if backend != "mlx":
            return [self.answer(messages)]
        return [self.answer(messages, model, tokenizer) for _, model, tokenizer in models]

    def answer(self, messages: list[dict[str, str]], model: Any = None, tokenizer: Any = None) -> str:
        """One model's reply to a chat, greedy, thinking off, under the GPU lock."""
        with self.lock:
            backend = self.backend
            model = model if model is not None else self.model
            tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        if backend == "mlx":
            import mlx.core as mx
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            with gpu_lock.LOCK, mx.stream(self.mlx_stream):
                text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_NEW_TOKENS, sampler=make_sampler(temp=0.0), verbose=False)
        else:
            import torch

            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False
            )
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
            text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        # A thinking model may reason first (Qwen's <think>…</think>, Gemma 4's
        # <|channel>thought … <channel|>); only what follows is the answer.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL)
        return text.strip()

    def suggest(
        self,
        text: str,
        vocabulary: list[str],
        examples: list[dict[str, str]] | None = None,
        learned: list[tuple[str, str]] | None = None,
        alternates: list[dict[str, str]] | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        self.wait_if_loading(20.0)
        with self.lock:
            model = self.model
        if model is None or not text.strip():
            return [], self.state
        started = time.time()
        messages = build_messages(text, examples or [], alternates)
        proposals = merge_proposals([parse_answer(answer) for answer in self.answers(messages)])
        taught = list(learned or []) + [(str(e.get("wrong", "")), str(e.get("right", ""))) for e in examples or []]
        kept = validate(proposals, text, vocabulary, taught, alternates)
        print(
            f"akıllı düzeltme {time.time() - started:.1f} s: önerilen {proposals!r} → uygulanan {kept!r}",
            file=sys.stderr,
            flush=True,
        )
        return kept, "ready"
