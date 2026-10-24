#!/usr/bin/env python3
"""Persistent local worker for the unmodified official Whisper large-v3 checkpoint.

Protocol: one JSON object per line on stdin/stdout. stdout is reserved for the
protocol; diagnostic text goes to stderr. The process loads the model once and
keeps it in memory until the native app's idle timeout closes this worker.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock  # noqa: E402  (sits next to this file in the app bundle)
import term_bias  # noqa: E402
import term_llm  # noqa: E402
import memory as memory_store  # noqa: E402

MODEL_ID = "openai/whisper-large-v3"
MODEL_URL = "https://huggingface.co/openai/whisper-large-v3"
SAMPLE_RATE = 16_000
EXPECTED_PARAMETER_COUNT = 1_543_490_560

# Last-resort escape from a repetition loop: re-decode a segment at a higher
# temperature when its output looks too repetitive. Whisper's reference
# implementation does this, and transformers carries the machinery — but
# `temperature` defaults to None, which collapses the ladder to a single pass
# and returns the looped text as-is. Supplying a ladder switches it on.
#
# This is a net, not the fix. The first rung is 0.0 (do_sample=False), so
# healthy audio never leaves the deterministic path: measured over the five
# archived recordings the ladder never fires and every run returns identical
# text. It matters only when a segment degenerates anyway, and then a sampled
# retry beats a phrase repeated thirty times.
#
# Keep it a LAST resort. The retries sample, so a rescued segment is not
# reproducible — when this ladder was carrying the fix on its own, the same
# recording returned three different wrong transcripts on three runs, one of
# which echoed this prompt back into the transcript. Anything that stops
# segments from degenerating in the first place belongs above this line.
TEMPERATURE_LADDER = (0.0, 0.4, 0.8)

# gzip ratio of a segment's tokens, above which the segment is re-decoded.
# Measured with transformers' own metric on this checkpoint: clean speech
# scores ~1.00 and a 31-repeat loop scores 3.15, so 1.35 fires well before a
# loop grows long while leaving normal dictation untouched.
COMPRESSION_RATIO_THRESHOLD = 1.35

# Deliberately NOT set: logprob_threshold and no_speech_threshold. Both fire on
# perfectly healthy short dictation and walk every clip up the whole ladder —
# the same 18.5 s recording takes 2.9 s without them and 21.7 s with them, for
# identical text. (Under transformers 4.46.3 no_speech_threshold without
# logprob_threshold also raised UnboundLocalError.)
#
# Deliberately NOT set either: num_beams. Beam search (5 beams, Whisper's own
# CLI default) was measured against greedy on the five archived recordings
# under white and pink noise down to 0 dB SNR, at -24/-36 dB gain and on a
# 23 s concatenation: greedy 8.1 % word errors, beam 10.3 %, and beam took
# 2.2x longer. Beam's extra errors were rewrites of what was said ("abi" →
# "ağabey", "sağolasın" → "sağ olasın"). fp32 was also tried: identical text
# to fp16, 40 % slower.

# The prompt (context sentence + term list) stays on every window, because it
# is what makes technical terms come out spelled right: on ten synthesized
# sentences it took term spelling from 11/20 to 20/20 and halved the word
# error rate. Its one measured failure mode is on audio longer than a single
# 30 s window: transformers prepends the prompt to every window, and on one
# 46 s recording the second window came back as a single subtitle artifact
# ("Bu videoyu izlediğiniz için teşekkür ederim.") spanning the whole window
# while 19 s of real speech vanished. Decoding the same audio without the
# prompt was word-perfect. So a long-form result is checked for a window that
# is long but nearly empty, and only then the audio is decoded again without
# the prompt. Healthy audio never pays for the second pass.
LONG_FORM_SECONDS = 30.0
SWALLOWED_WINDOW_SECONDS = 12.0
SWALLOWED_WORDS_PER_SECOND = 0.5

# Contextual biasing toward the user's terms and the bundled software
# vocabulary, single-window audio only; strength and the reasons are in
# term_bias.py. The trie is rebuilt only when the term list changes.
_bias_cache: dict[tuple[str, ...], Any] = {}


def emit(message: dict[str, Any]) -> None:
    print(json.dumps(message, ensure_ascii=False), flush=True)


def emit_state(state: str, detail: str = "") -> None:
    emit({"event": "state", "state": state, "detail": detail})


def fail_fatally(error: BaseException) -> None:
    emit(
        {
            "event": "fatal",
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    )


def content_address(weights: Path) -> str | None:
    """The SHA-256 Hugging Face used to name this file's blob, or None.

    The snapshot entry is a symlink into `blobs/<sha256>`. Since huggingface-hub
    2.x that blob may itself be a symlink into a shared, content-defined store
    whose file names are Xet hashes — also 64 hex characters, but not the
    SHA-256 of the file. Following every link to the end therefore picks up the
    wrong name and makes a healthy download fail verification; the first hop is
    the one that carries the SHA-256.
    """
    current = weights
    for _ in range(8):
        if len(current.name) == 64 and all(
            character in "0123456789abcdef" for character in current.name
        ):
            return current.name
        if not current.is_symlink():
            return None
        current = current.parent / Path(os.readlink(current))
    return None


def verify_weights(weights: Path, snapshot: Path) -> tuple[str, int]:
    """Verify once, then reuse a file-identity marker on later cold starts."""
    size_bytes = weights.stat().st_size
    if size_bytes < 3_000_000_000:
        raise RuntimeError(
            f"model.safetensors beklenenden küçük ({size_bytes} bayt); tam large-v3 doğrulanamadı."
        )

    expected_sha256 = content_address(weights)
    if expected_sha256 is None:
        raise RuntimeError("Hugging Face içerik adresli safetensors özeti bulunamadı.")

    resolved = weights.resolve()
    identity = {
        "snapshot_revision": snapshot.name,
        "weights_path": str(resolved),
        "weights_sha256": expected_sha256,
        "weights_bytes": size_bytes,
        "weights_mtime_ns": resolved.stat().st_mtime_ns,
    }
    marker_root = Path(os.environ.get("HF_HOME", snapshot.parent)) / "whisperhold-verification"
    marker = marker_root / f"{snapshot.name}-{expected_sha256}.json"
    try:
        if json.loads(marker.read_text(encoding="utf-8")) == identity:
            emit_state("checking", "Daha önce doğrulanan resmî safetensors bulundu…")
            return expected_sha256, size_bytes
    except (FileNotFoundError, OSError, ValueError, TypeError):
        pass

    emit_state("checking", "Resmî safetensors SHA-256 özeti doğrulanıyor…")
    digest = hashlib.sha256()
    with weights.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "model.safetensors SHA-256 doğrulaması başarısız: "
            f"beklenen={expected_sha256} bulunan={actual_sha256}"
        )

    marker_root.mkdir(parents=True, exist_ok=True)
    temporary_marker = marker.with_suffix(f".{os.getpid()}.tmp")
    temporary_marker.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    temporary_marker.replace(marker)
    return actual_sha256, size_bytes


def load_runtime() -> tuple[Any, Any, Any, str, str, dict[str, Any]]:
    # Imports happen after the first state event so the native UI never looks
    # frozen while PyTorch itself is being imported.
    emit_state("checking", "Python ve PyTorch denetleniyor…")
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    # Only skip duplicate framework formats (pytorch_model.bin/flax/tf). The
    # exact safetensors checkpoint published by OpenAI is downloaded as-is.
    allowed_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "normalizer.json",
        "merges.txt",
        "vocab.json",
    ]
    try:
        snapshot_path = snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=allowed_files,
            local_files_only=True,
        )
        emit_state("checking", "Resmî model yerel önbellekte bulundu…")
    except Exception:
        emit_state(
            "downloading",
            "Resmî openai/whisper-large-v3 dosyaları denetleniyor/indiriliyor…",
        )
        snapshot_path = snapshot_download(repo_id=MODEL_ID, allow_patterns=allowed_files)
    snapshot = Path(snapshot_path).resolve()
    weights = snapshot / "model.safetensors"
    config_file = snapshot / "config.json"
    if not weights.is_file() or not config_file.is_file():
        raise RuntimeError("Resmî snapshot içinde model.safetensors veya config.json bulunamadı.")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("model_type") != "whisper":
        raise RuntimeError(f"Beklenmeyen model_type: {config.get('model_type')!r}")
    if config.get("encoder_layers") != 32 or config.get("decoder_layers") != 32:
        raise RuntimeError(
            "İndirilen yapı tam large-v3 değil: "
            f"encoder={config.get('encoder_layers')} decoder={config.get('decoder_layers')}"
        )

    actual_sha256, size_bytes = verify_weights(weights, snapshot)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    emit_state("loading", f"Tam model belleğe yükleniyor ({device.upper()})…")
    processor = AutoProcessor.from_pretrained(snapshot_path, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        snapshot_path,
        local_files_only=True,
        use_safetensors=True,
        dtype=dtype,
    )
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Tam large-v3 parametre sayısı uyuşmuyor: "
            f"beklenen={EXPECTED_PARAMETER_COUNT} bulunan={parameter_count}"
        )
    # Dil/görev her dikte isteğinde açıkça veriliyor; checkpoint'teki eski
    # forced_decoder_ids ile iki kez verilmesini ve olası çatışmayı önle.
    model.generation_config.forced_decoder_ids = None
    # pad_token_id ile eos_token_id aynı olduğundan attention mask açıkça
    # üretilmeli; aksi hâlde Transformers güvenilirlik uyarısı veriyor.
    processor.feature_extractor.return_attention_mask = True

    # The model is driven through `generate` directly, not through transformers'
    # automatic-speech-recognition pipeline. The pipeline was dropped after a
    # measurement: from transformers 4.5x on it silently applies its own
    # generation defaults whenever the checkpoint does not set them —
    # `num_beams=5` and `max_new_tokens=256` — and this checkpoint sets
    # neither. Every dictation was then decoded with beam search, which on
    # these recordings had already been measured as slower (2–3x) and less
    # faithful ("abi" → "ağabey"), and every 30 s window was capped at 256
    # tokens. Calling `generate` ourselves leaves no hidden defaults.
    #
    # No chunking either: Whisper's own sequential long-form algorithm handles
    # audio longer than 30 s. The difference that matters is where the cut
    # falls. Chunking slices at a fixed step regardless of what is being said,
    # decodes each slice independently and stitches by overlap; a sentence
    # straddling a cut is decoded twice from two half-contexts. Sequential
    # decoding advances to the end of the last segment the model itself
    # closed, so a cut lands between segments rather than inside one. On one
    # 50 s recording the chunked path turned a disfluent stretch into the same
    # phrase repeated 31 times and swallowed the real speech after it;
    # sequential decodes it correctly and identically on repeated runs.
    #
    # It does NOT feed the previous window's text to the next one: that is
    # `condition_on_prev_tokens`, which stays off here. Turning it on would
    # carry a bad window's text forward into the next, a different trade.
    with gpu_lock.LOCK:
        model.to(device)

    verification = {
        "model_id": MODEL_ID,
        "source_url": MODEL_URL,
        "snapshot_path": str(snapshot),
        "snapshot_revision": snapshot.name,
        "weights_path": str(weights),
        "weights_sha256": actual_sha256,
        "weights_bytes": size_bytes,
        "encoder_layers": config["encoder_layers"],
        "decoder_layers": config["decoder_layers"],
        "d_model": config.get("d_model"),
        "parameter_count": parameter_count,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }
    runtime = {"model": model, "dtype_torch": dtype, "vocabulary": term_bias.load_vocabulary(), **verification}
    return np, torch, processor, device, snapshot_path, runtime


def make_prompt(processor: Any, prompt: str, torch: Any, device: str) -> Any | None:
    cleaned = prompt.strip()
    if not cleaned:
        return None
    prompt_ids = processor.get_prompt_ids(cleaned, return_tensors="pt")
    # Whisper leaves at most 223 prompt tokens. Keep the most recent terms and
    # context when the user wrote a longer paragraph.
    #
    # Index 0 is the <|startofprev|> marker and has to survive the cut. It is
    # what tells the decoder where the prompt ends, and transformers strips the
    # prompt from the returned text by looking for exactly that token
    # (tokenization_whisper.py `_strip_prompt`). A plain [-223:] drops it, and
    # then the user's whole context sentence and vocabulary list are decoded as
    # if they had been spoken and pasted in front of every transcript.
    if prompt_ids.shape[-1] > 223:
        prompt_ids = torch.cat([prompt_ids[..., :1], prompt_ids[..., -222:]], dim=-1)
    return prompt_ids.to(torch.device(device))


def swallowed_window(segments: list[dict[str, Any]], processor: Any) -> str | None:
    """Describe the first segment that is long but nearly empty.

    A real sentence runs at two to four words per second; a 30 s window
    holding seven words is the signature of the model having replaced the
    window with an artifact. A long pause followed by one short sentence can
    look the same — that case only costs a second, prompt-free pass that
    returns the same sentence.
    """
    for segment in segments:
        start, end = segment.get("start"), segment.get("end")
        if start is None or end is None:
            continue
        duration = float(end) - float(start)
        text = processor.tokenizer.decode(segment["tokens"], skip_special_tokens=True)
        words = len(text.split())
        if duration >= SWALLOWED_WINDOW_SECONDS and words < duration * SWALLOWED_WORDS_PER_SECOND:
            return f"{duration:.1f} s pencerede {words} kelime"
    return None


def bias_processor(torch: Any, processor: Any, terms: list[str]) -> Any | None:
    key = tuple(terms)
    if not key:
        return None
    if key not in _bias_cache:
        _bias_cache.clear()
        _bias_cache[key] = term_bias.make_bias_processor(torch, processor.tokenizer, list(key))
    return _bias_cache[key]


def decode(
    audio: Any, generate_kwargs: dict[str, Any], torch: Any, processor: Any, runtime: dict[str, Any],
    bias_terms: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """One pass of the model over the whole recording: text and its segments."""
    model = runtime["model"]
    device = runtime["device"]
    long_form = audio.size > LONG_FORM_SECONDS * SAMPLE_RATE
    if bias_terms and not long_form:
        from transformers import LogitsProcessorList

        generate_kwargs = dict(generate_kwargs)
        generate_kwargs["logits_processor"] = LogitsProcessorList([bias_processor(torch, processor, bias_terms)])
    if long_form:
        # Longer than one window: hand the whole recording over and let the
        # sequential algorithm walk it. The attention mask tells it where the
        # audio really ends.
        inputs = processor.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
    else:
        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
        )
    # The GPU lock: the smart-correction and embedding models load on other
    # threads and must not touch Metal while a dictation is being decoded.
    with gpu_lock.LOCK, torch.inference_mode():
        features = inputs.input_features.to(device, runtime["dtype_torch"])
        attention_mask = inputs.attention_mask.to(device)
        output = model.generate(
            features,
            attention_mask=attention_mask,
            return_timestamps=True,
            return_segments=True,
            **generate_kwargs,
        )
    sequences = output["sequences"]
    segments = output["segments"][0] if output.get("segments") else []
    # skip_special_tokens drops the timestamp tokens and, through the
    # tokenizer's prompt stripping, the prompt itself.
    text = processor.tokenizer.decode(sequences[0], skip_special_tokens=True)
    return str(text).strip(), segments


def build_prompt(request: dict[str, Any], processor: Any, runtime: dict[str, Any]) -> str:
    """The prompt for this dictation.

    New-style requests carry the context sentence, the hand-written terms and
    the learned terms separately; memory composes them into the token budget,
    choosing the learned terms this dictation most likely needs. A request
    that only carries `prompt` (older app builds, or the no-prompt retry) is
    used as is.
    """
    if "context" not in request and "contexts" not in request and "prompt_terms" not in request:
        return str(request.get("prompt") or "")
    contexts = [str(c) for c in request.get("contexts") or [] if str(c).strip()]
    if not contexts and str(request.get("context") or "").strip():
        contexts = [line.strip() for line in str(request["context"]).splitlines() if line.strip()]
    prompt_terms = [str(term) for term in request.get("prompt_terms") or [] if str(term).strip()]
    learned = [str(term) for term in request.get("learned_terms") or [] if str(term).strip()]
    store = runtime.get("memory")
    if store is None:
        all_terms = prompt_terms + learned
        context = " ".join(contexts)
        if not all_terms:
            return context
        return (context + " " if context else "") + "Özel adlar ve teknik terimler aynen şu yazımla kullanılır: " + ", ".join(all_terms) + "."

    # The hand-written list and the context sentences live in memory too, so
    # they can grow without limit; the request carries the current versions.
    if "prompt_terms" in request:
        store.set_vocabulary(prompt_terms)
    if "contexts" in request or "context" in request:
        store.set_contexts(contexts)

    def count_tokens(text: str) -> int:
        return int(len(processor.get_prompt_ids(text))) if text.strip() else 0

    prompt, chosen = store.compose_prompt(contexts, prompt_terms, learned, count_tokens)
    print(
        f"prompt: {len(chosen)}/{len(prompt_terms) + len(learned)} terim ({len(prompt_terms)} yazılı, {len(learned)} öğrenilen), "
        f"{len(contexts)} bağlam cümlesi, {count_tokens(prompt)} token",
        file=sys.stderr,
        flush=True,
    )
    return prompt


def transcribe(
    request: dict[str, Any], np: Any, torch: Any, processor: Any, runtime: dict[str, Any]
) -> str:
    audio_path = Path(str(request["audio_path"]))
    audio = np.fromfile(audio_path, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Ses dosyası boş.")
    if not np.isfinite(audio).all():
        raise ValueError("Ses örneklerinde geçersiz değer var.")

    language = str(request.get("language") or "auto")
    prompt = build_prompt(request, processor, runtime)
    generate_kwargs: dict[str, Any] = {
        "task": "transcribe",
        # Greedy: beam search was measured as slower and less faithful here,
        # see the module comment. Spelled out so no library default can
        # change it underneath us again.
        "num_beams": 1,
        "temperature": TEMPERATURE_LADDER,
        "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
    }
    if language != "auto":
        generate_kwargs["language"] = language
    with gpu_lock.LOCK:
        prompt_ids = make_prompt(processor, prompt, torch, runtime["device"])
    if prompt_ids is not None:
        generate_kwargs["prompt_ids"] = prompt_ids

    # Biasing vocabulary: the user's own terms first, then the bundled list
    # when the app asks for it.
    bias_terms = [str(term) for term in (request.get("bias_terms") or request.get("terms") or []) if str(term).strip()]
    if request.get("builtin_vocabulary"):
        seen = set(bias_terms)
        bias_terms += [term for term in runtime["vocabulary"] if term not in seen]

    text, segments = decode(audio, generate_kwargs, torch, processor, runtime, bias_terms)

    if prompt_ids is not None and audio.size > LONG_FORM_SECONDS * SAMPLE_RATE:
        reason = swallowed_window(segments, processor)
        if reason is not None:
            print(f"uzun kayıtta yutulmuş pencere ({reason}); prompt'suz yeniden çözümleniyor", file=sys.stderr, flush=True)
            retry_kwargs = {key: value for key, value in generate_kwargs.items() if key != "prompt_ids"}
            retry_text, retry_segments = decode(audio, retry_kwargs, torch, processor, runtime, bias_terms)
            if swallowed_window(retry_segments, processor) is None:
                text = retry_text

    return text


def main() -> int:
    emit({"event": "boot", "model_id": MODEL_ID, "pid": os.getpid()})
    try:
        np, torch, processor, device, snapshot_path, runtime = load_runtime()
    except BaseException as error:
        fail_fatally(error)
        return 1

    emit({
        "event": "ready",
        "vocabulary_terms": len(runtime["vocabulary"]),
        **{key: value for key, value in runtime.items() if key not in ("model", "dtype_torch", "vocabulary")},
    })

    smart = term_llm.SmartCorrector(emit)
    support_root = Path(os.environ.get("HF_HOME") or (Path.home() / "Library/Application Support/WhisperHold/HuggingFace")).parent
    try:
        store: memory_store.Memory | None = memory_store.Memory(support_root / "memory.sqlite", emit)
    except Exception as error:  # noqa: BLE001 - memory is an aid, never a blocker
        print(f"hafıza açılamadı: {error}", file=sys.stderr, flush=True)
        store = None
    runtime["memory"] = store

    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            request_id = str(request.get("id", ""))
            command = request.get("command")
            if command == "transcribe":
                emit({"event": "transcribing", "id": request_id})
                text = transcribe(request, np, torch, processor, runtime)
                emit({"event": "result", "id": request_id, "text": text})
            elif command == "load_llm":
                # Akıllı düzeltme açıkken uygulama bunu worker her başladığında
                # gönderir; model arka planda yüklenir, diktenin önünü kesmez.
                smart.start()
                emit({"event": "llm_state", "id": request_id, "state": smart.state, "detail": smart.detail})
            elif command == "suggest":
                terms = [str(term) for term in request.get("terms") or [] if str(term).strip()]
                vocabulary = terms + [term for term in runtime["vocabulary"] if term not in set(terms)]
                text = str(request.get("text") or "")
                examples = [example for example in request.get("examples") or [] if isinstance(example, dict)]
                if store is not None:
                    seen = {(str(e.get("wrong")), str(e.get("right"))) for e in examples}
                    for example in store.nearest_corrections(text):
                        if (example["wrong"], example["right"]) not in seen:
                            examples.append(example)
                replacements, state = smart.suggest(text, vocabulary, examples)
                emit({"event": "suggestions", "id": request_id, "replacements": replacements, "state": state})
            elif command == "load_memory":
                if store is not None:
                    store.start_embedder()
                else:
                    emit({"event": "memory_state", "embedder": "failed", "detail": "Hafıza açılamadı"})
            elif command == "remember":
                if store is not None:
                    kind = request.get("kind")
                    if kind == "dictation":
                        store.remember_dictation(str(request.get("text") or ""), str(request.get("app") or ""),
                                                 [str(t) for t in request.get("terms") or []])
                    elif kind == "correction":
                        store.remember_correction(str(request.get("wrong") or ""), str(request.get("right") or ""),
                                                  str(request.get("context") or ""), bool(request.get("safe")))
                    emit({"event": "memory_state", **store.stats()})
            elif command == "forget":
                if store is not None:
                    store.forget_correction(str(request.get("wrong") or ""), str(request.get("right") or ""))
                    emit({"event": "memory_state", **store.stats()})
            elif command == "memory_clear":
                if store is not None:
                    store.clear()
                    emit({"event": "memory_state", **store.stats()})
            elif command == "memory_stats":
                if store is not None:
                    emit({"event": "memory_state", **store.stats()})
            elif command == "ping":
                emit({"event": "pong", "id": request_id})
            elif command == "shutdown":
                emit({"event": "stopped", "id": request_id})
                return 0
            else:
                raise ValueError(f"Bilinmeyen komut: {command!r}")
        except BaseException as error:
            emit(
                {
                    "event": "error",
                    "id": str(locals().get("request_id", "")),
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
