# ─────────────────────────────────────────────────────────────────────────────
# wer_eval/config.py  —  Central configuration
# Edit DEFAULTS for persistent changes; all keys are overridable via CLI.
# ─────────────────────────────────────────────────────────────────────────────

import argparse

DEFAULTS = {
    # ── Backend ───────────────────────────────────────────────────────────────
    # whisper      → openai/whisper-* family via HuggingFace transformers
    # hf_pipeline  → any AutoModelForSpeechSeq2Seq (e.g. wav2vec2, MMS, etc.)
    # json         → load pre-computed hypotheses from JSON files (no inference)
    "backend": "whisper",

    # ── Model ─────────────────────────────────────────────────────────────────
    "model_id": "openai/whisper-small",

    # ── Dataset ───────────────────────────────────────────────────────────────
    "dataset_id":    "palli23/spjallromur-4h",
    "split":         "train[:100]",
    "audio_col":     "audio",
    "text_col":      "text",        # raw reference
    "norm_text_col": "norm_text",   # normalised reference; used when present

    # ── Audio ─────────────────────────────────────────────────────────────────
    "sample_rate": 16000,

    # ── JSON backend ──────────────────────────────────────────────────────────
    # Directory of JSON files; each file: {"hypothesis": "...", "reference": "..."}
    # If "reference" is absent the dataset reference column is used instead.
    "json_dir": None,

    # ── Generation ────────────────────────────────────────────────────────────
    "language":       "is",         # ISO-639-1; set "none" to let model decide
    "max_new_tokens": 225,
    "batch_size":     8,

    # ── Output ────────────────────────────────────────────────────────────────
    "results_json": "wer_results.json",
    "results_csv":  "wer_results.csv",
    "plot_png":     "wer_vs_duration.png",
    "n_examples":   10,             # ref/hyp pairs printed in summary
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WER Evaluation Harness — Málrómur / Spjallrómur",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--backend", default=DEFAULTS["backend"],
                   choices=["whisper", "hf_pipeline", "json"])
    p.add_argument("--model-id",        default=DEFAULTS["model_id"])
    p.add_argument("--dataset",         default=DEFAULTS["dataset_id"], dest="dataset_id")
    p.add_argument("--split",           default=DEFAULTS["split"])
    p.add_argument("--audio-col",       default=DEFAULTS["audio_col"])
    p.add_argument("--text-col",        default=DEFAULTS["text_col"])
    p.add_argument("--norm-text-col",   default=DEFAULTS["norm_text_col"])
    p.add_argument("--sample-rate",     default=DEFAULTS["sample_rate"], type=int)
    p.add_argument("--json-dir",        default=DEFAULTS["json_dir"],
                   help="Directory of pre-computed JSON transcripts (json backend)")
    p.add_argument("--language",        default=DEFAULTS["language"])
    p.add_argument("--max-new-tokens",  default=DEFAULTS["max_new_tokens"], type=int)
    p.add_argument("--batch-size",      default=DEFAULTS["batch_size"],     type=int)
    p.add_argument("--results-json",    default=DEFAULTS["results_json"])
    p.add_argument("--results-csv",     default=DEFAULTS["results_csv"])
    p.add_argument("--plot",            default=DEFAULTS["plot_png"],  dest="plot_png")
    p.add_argument("--n-examples",      default=DEFAULTS["n_examples"],     type=int)
    p.add_argument("--hf-token",        default=None,
                   help="HuggingFace token for gated repos")
    p.add_argument("--device",          default=None,
                   help="Force device: cuda / cpu (default: auto-detect)")

    return p.parse_args()
