# ─────────────────────────────────────────────────────────────────────────────
# wer_eval/backends.py  —  ASR backend abstraction
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


# ── Protocol ─────────────────────────────────────────────────────────────────

class ASRBackend(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...
    @property
    def name(self) -> str: ...


# ── Device helpers ────────────────────────────────────────────────────────────

def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Whisper backend ───────────────────────────────────────────────────────────

# Samples longer than this (in seconds) require return_timestamps=True
_WHISPER_MAX_S = 29.0

class WhisperBackend:
    """
    Runs any openai/whisper-* or fine-tuned Whisper checkpoint.
    Handles both short (<30s) and long-form audio transparently.
    """

    def __init__(
        self,
        model_id: str,
        language: str | None = "is",
        max_new_tokens: int = 225,
        device: str | None = None,
        hf_token: str | None = None,
    ):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self._model_id = model_id
        self._device   = _resolve_device(device)
        torch_dtype    = torch.float16 if self._device == "cuda" else torch.float32

        print(f"[whisper] Loading {model_id} on {self._device} …")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=hf_token,
        ).to(self._device)

        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

        self._gen_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if language and language.lower() != "none":
            self._gen_kwargs["language"] = language

        # One pipeline for short clips (no timestamps)
        self._pipe_short = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self._device,
            generate_kwargs=self._gen_kwargs,
        )

        # One pipeline for long clips (timestamps required by Whisper)
        self._pipe_long = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self._device,
            generate_kwargs=self._gen_kwargs,
            return_timestamps=True,
        )

    @property
    def name(self) -> str:
        return f"whisper:{self._model_id}"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        duration = len(audio) / sample_rate
        pipe = self._pipe_long if duration > _WHISPER_MAX_S else self._pipe_short

        # Suppress the timestamp warning — we handle it explicitly
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*ending timestamp.*")
            warnings.filterwarnings("ignore", message=".*WhisperTimestamp.*")
            result = pipe({"array": audio, "sampling_rate": sample_rate})

        return result["text"].strip()


# ── Generic HF pipeline backend ───────────────────────────────────────────────

class HFPipelineBackend:
    """
    Wraps any model compatible with the 'automatic-speech-recognition' pipeline.
    """

    def __init__(
        self,
        model_id: str,
        language: str | None = None,
        max_new_tokens: int = 225,
        device: str | None = None,
        hf_token: str | None = None,
    ):
        from transformers import pipeline

        self._model_id = model_id
        self._device   = _resolve_device(device)

        print(f"[hf_pipeline] Loading {model_id} on {self._device} …")

        gen_kwargs: dict = {}
        if max_new_tokens:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if language and language.lower() != "none":
            gen_kwargs["language"] = language

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if self._device == "cuda" else -1,
            generate_kwargs=gen_kwargs or None,
            token=hf_token,
        )

    @property
    def name(self) -> str:
        return f"hf_pipeline:{self._model_id}"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        result = self._pipe({"array": audio, "sampling_rate": sample_rate})
        return result["text"].strip()


# ── JSON backend ──────────────────────────────────────────────────────────────

class JSONBackend:
    """
    Loads pre-computed hypotheses from a directory of JSON files.
    Naming: <zero_padded_index>.json  e.g. 00000.json
    Each file: {"hypothesis": "...", "reference": "..."}  (reference optional)
    """

    def __init__(self, json_dir: str):
        self._dir   = Path(json_dir)
        self._cache: dict[int, dict] = {}
        self._index = 0

        if not self._dir.exists():
            raise FileNotFoundError(f"json_dir not found: {json_dir}")

        files = sorted(self._dir.glob("*.json"))
        if not files:
            raise ValueError(f"No JSON files found in {json_dir}")

        print(f"[json] Found {len(files)} hypothesis files in {json_dir}")
        for f in files:
            idx = int(f.stem)
            with open(f) as fh:
                self._cache[idx] = json.load(fh)

    @property
    def name(self) -> str:
        return f"json:{self._dir.name}"

    def get_reference_override(self, idx: int) -> str | None:
        return self._cache.get(idx, {}).get("reference")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        entry = self._cache.get(self._index, {})
        self._index += 1
        hyp = entry.get("hypothesis", "")
        if not hyp:
            print(f"  ⚠  No hypothesis for index {self._index - 1}")
        return hyp.strip()


# ── Factory ───────────────────────────────────────────────────────────────────

def build_backend(cfg) -> ASRBackend:
    if cfg.backend == "whisper":
        return WhisperBackend(
            model_id=cfg.model_id,
            language=cfg.language,
            max_new_tokens=cfg.max_new_tokens,
            device=getattr(cfg, "device", None),
            hf_token=getattr(cfg, "hf_token", None),
        )
    elif cfg.backend == "hf_pipeline":
        return HFPipelineBackend(
            model_id=cfg.model_id,
            language=cfg.language,
            max_new_tokens=cfg.max_new_tokens,
            device=getattr(cfg, "device", None),
            hf_token=getattr(cfg, "hf_token", None),
        )
    elif cfg.backend == "json":
        if not cfg.json_dir:
            raise ValueError("--json-dir is required for the json backend")
        return JSONBackend(json_dir=cfg.json_dir)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")