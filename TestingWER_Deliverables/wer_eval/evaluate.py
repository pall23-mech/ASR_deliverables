# ─────────────────────────────────────────────────────────────────────────────
# wer_eval/evaluate.py  —  Dataset loading and evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .backends import ASRBackend, JSONBackend
from .metrics  import EditStats, sample_wer, sample_cer
from .normalise import normalise
from .report   import print_summary, save_csv, save_json, save_plot


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(cfg) -> Any:
    from datasets import load_dataset as hf_load, Audio

    print(f"[dataset] Loading {cfg.dataset_id}  split={cfg.split} …")
    ds = hf_load(
        cfg.dataset_id,
        split=cfg.split,
        token=getattr(cfg, "hf_token", None),
    )
    ds = ds.cast_column(cfg.audio_col, Audio(sampling_rate=cfg.sample_rate))
    print(f"[dataset] {len(ds)} samples loaded")
    return ds


def _get_audio(sample: dict, audio_col: str) -> np.ndarray:
    arr = sample[audio_col]["array"]
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.astype(np.float32)


def _get_reference(sample: dict, cfg, idx: int, backend: ASRBackend) -> str:
    # JSON backend may carry its own reference
    if isinstance(backend, JSONBackend):
        override = backend.get_reference_override(idx)
        if override:
            return normalise(override)

    raw_ref  = sample.get(cfg.text_col, "") or ""
    norm_ref = sample.get(cfg.norm_text_col, "") or ""
    return normalise(raw_ref, norm_ref if norm_ref else None)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_evaluation(cfg) -> None:
    from .backends import build_backend

    backend = build_backend(cfg)
    ds      = load_dataset(cfg)

    results:    list[dict] = []
    word_stats  = EditStats()
    cer_total   = 0.0
    n_skipped   = 0

    print(f"\n[eval] Running {len(ds)} samples with backend: {backend.name}\n")

    for idx, sample in enumerate(ds):
        audio = _get_audio(sample, cfg.audio_col)
        duration_s = len(audio) / cfg.sample_rate

        # Transcribe
        t0 = time.perf_counter()
        try:
            hyp = backend.transcribe(audio, cfg.sample_rate)
        except Exception as exc:
            print(f"  ⚠  idx={idx}: transcription failed — {exc}")
            hyp = ""
        elapsed = time.perf_counter() - t0

        ref = _get_reference(sample, cfg, idx, backend)

        # Skip if either side is empty
        if not ref or not hyp:
            n_skipped += 1
            results.append({
                "idx": idx, "duration_s": round(duration_s, 2),
                "reference": ref, "hypothesis": hyp,
                "wer": None, "cer": None,
                "hits": None, "substitutions": None,
                "deletions": None, "insertions": None,
                "ref_tokens": None, "rtf": None,
                "skipped": True,
            })
            continue

        stats   = sample_wer(ref, hyp)
        cer_val = sample_cer(ref, hyp)
        rtf     = elapsed / duration_s if duration_s > 0 else None

        word_stats.add(
            stats["hits"], stats["substitutions"],
            stats["deletions"], stats["insertions"],
            stats["ref_tokens"],
        )
        cer_total += cer_val

        results.append({
            "idx":           idx,
            "duration_s":    round(duration_s, 2),
            "reference":     ref,
            "hypothesis":    hyp,
            "wer":           stats["wer"],
            "cer":           cer_val,
            "hits":          stats["hits"],
            "substitutions": stats["substitutions"],
            "deletions":     stats["deletions"],
            "insertions":    stats["insertions"],
            "ref_tokens":    stats["ref_tokens"],
            "rtf":           round(rtf, 4) if rtf else None,
            "skipped":       False,
        })

        if (idx + 1) % 10 == 0:
            scored = idx + 1 - n_skipped
            running_wer = word_stats.wer * 100 if word_stats.ref_len else 0.0
            print(f"  {idx+1:>5}/{len(ds)}  WER so far: {running_wer:.2f}%  "
                  f"(skipped: {n_skipped})")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    scored_results = [r for r in results if not r["skipped"]]
    n_scored = len(scored_results)

    summary = {
        "backend":       backend.name,
        "dataset":       cfg.dataset_id,
        "split":         cfg.split,
        "n_samples":     len(ds),
        "n_scored":      n_scored,
        "n_skipped":     n_skipped,
        **word_stats.summary(),
        "cer":           round(cer_total / n_scored, 2) if n_scored else 0.0,
    }

    print_summary(summary, scored_results, n_examples=cfg.n_examples)
    save_json(results, summary, cfg.results_json)
    save_csv(results,           cfg.results_csv)
    save_plot(scored_results,   cfg.plot_png)
