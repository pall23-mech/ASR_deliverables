#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# wer_eval.py  —  Entry point
#
# Usage:
#   python wer_eval.py                                        # Whisper defaults
#   python wer_eval.py --backend whisper --model-id openai/whisper-large-v3
#   python wer_eval.py --backend hf_pipeline --model-id facebook/mms-300m
#   python wer_eval.py --backend json --json-dir ./my_transcripts/
#   python wer_eval.py --split "train[:20]" --n-examples 5
# ─────────────────────────────────────────────────────────────────────────────

from wer_eval.config   import parse_args
from wer_eval.evaluate import run_evaluation

if __name__ == "__main__":
    cfg = parse_args()
    run_evaluation(cfg)
