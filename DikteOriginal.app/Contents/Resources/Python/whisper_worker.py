#!/usr/bin/env python3
"""Persistent local worker for the unmodified official Whisper large-v3 checkpoint.

Protocol: one JSON object per line on stdin/stdout. stdout is reserved for the
protocol; diagnostic text goes to stderr. The process loads the model once and
keeps it in memory between dictations.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

MODEL_ID = "openai/whisper-large-v3"
MODEL_URL = "https://huggingface.co/openai/whisper-large-v3"
SAMPLE_RATE = 16_000
EXPECTED_PARAMETER_COUNT = 1_543_490_560


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


def load_runtime() -> tuple[Any, Any, Any, str, str, dict[str, Any]]:
    # Imports happen after the first state event so the native UI never looks
    # frozen while PyTorch itself is being imported.
    emit_state("checking", "Python ve PyTorch denetleniyor…")
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    emit_state(
        "downloading",
        "Resmî openai/whisper-large-v3 dosyaları denetleniyor/indiriliyor…",
    )

    # Only skip duplicate framework formats (pytorch_model.bin/flax/tf). The
    # exact safetensors checkpoint published by OpenAI is downloaded as-is.
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=[
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
        ],
    )
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

    size_bytes = weights.stat().st_size
    if size_bytes < 3_000_000_000:
        raise RuntimeError(
            f"model.safetensors beklenenden küçük ({size_bytes} bayt); tam large-v3 doğrulanamadı."
        )

    emit_state("checking", "Resmî safetensors SHA-256 özeti doğrulanıyor…")
    expected_sha256 = weights.resolve().name
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise RuntimeError("Hugging Face içerik adresli safetensors özeti bulunamadı.")
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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    emit_state("loading", f"Tam model belleğe yükleniyor ({device.upper()})…")
    processor = AutoProcessor.from_pretrained(snapshot_path, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        snapshot_path,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
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

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=28,
        stride_length_s=(4, 2),
        torch_dtype=dtype,
        device=device,
    )

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
    return np, torch, processor, device, snapshot_path, {"asr": asr, **verification}


def make_prompt(processor: Any, prompt: str, torch: Any, device: str) -> Any | None:
    cleaned = prompt.strip()
    if not cleaned:
        return None
    prompt_ids = processor.get_prompt_ids(cleaned, return_tensors="pt")
    # Whisper leaves at most 223 prompt tokens. Keep the most recent terms and
    # context when the user wrote a longer paragraph.
    if prompt_ids.shape[-1] > 223:
        prompt_ids = prompt_ids[..., -223:]
    return prompt_ids.to(torch.device(device))


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
    prompt = str(request.get("prompt") or "")
    generate_kwargs: dict[str, Any] = {"task": "transcribe"}
    if language != "auto":
        generate_kwargs["language"] = language
    prompt_ids = make_prompt(processor, prompt, torch, runtime["device"])
    if prompt_ids is not None:
        generate_kwargs["prompt_ids"] = prompt_ids

    result = runtime["asr"](
        audio,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )
    return str(result.get("text", "")).strip()


def main() -> int:
    emit({"event": "boot", "model_id": MODEL_ID, "pid": os.getpid()})
    try:
        np, torch, processor, device, snapshot_path, runtime = load_runtime()
    except BaseException as error:
        fail_fatally(error)
        return 1

    emit({"event": "ready", **{key: value for key, value in runtime.items() if key != "asr"}})

    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            request_id = str(request.get("id", ""))
            command = request.get("command")
            if command == "transcribe":
                emit({"event": "transcribing", "id": request_id})
                text = transcribe(request, np, torch, processor, runtime)
                emit({"event": "result", "id": request_id, "text": text})
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
