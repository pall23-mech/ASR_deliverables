"""
build_spjallromur_dataset.py
Segment-aware conversational dataset builder

Target chunk length ≈ 24 seconds
Hard maximum chunk length = 40 seconds
Never drops audio
Never cuts segments unless segment > 40 seconds
"""

import argparse
import json
import logging
import os
import random
from math import gcd
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Time parsing
# ──────────────────────────────────────────────

def parse_time(t) -> Optional[float]:
    try:
        return float(str(t).replace("s", "").strip())
    except Exception:
        return None


# ──────────────────────────────────────────────
# Load transcript segments
# ──────────────────────────────────────────────

def load_segments(json_path: Path, speaker_label: str) -> List[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = []

    for seg in data.get("segments", []):
        start = parse_time(seg.get("startTime"))
        end = parse_time(seg.get("endTime"))

        if start is None or end is None or end <= start:
            continue

        text = " ".join(
            w.get("word", "").strip()
            for w in seg.get("words", [])
        ).strip()

        if not text:
            continue

        segments.append({
            "start": start,
            "end": end,
            "speaker": speaker_label,
            "text": text
        })

    return segments


# ──────────────────────────────────────────────
# Audio loading + resampling
# ──────────────────────────────────────────────

def load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        from scipy.signal import resample_poly
        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

    return audio


# ──────────────────────────────────────────────
# Segment-aware chunking
# ──────────────────────────────────────────────

def build_chunks(audio_a,
                 audio_b,
                 segs_a,
                 segs_b,
                 target_chunk_sec,
                 hard_max_sec,
                 sr):

    all_segs = sorted(segs_a + segs_b, key=lambda s: s["start"])

    chunks = []

    current_audio = []
    current_texts = []
    current_samples = 0

    target_samples = int(target_chunk_sec * sr)
    hard_max_samples = int(hard_max_sec * sr)

    for seg in all_segs:

        src = audio_a if seg["speaker"] == "A" else audio_b

        s = int(round(seg["start"] * sr))
        e = min(int(round(seg["end"] * sr)), len(src))

        if e <= s:
            continue

        seg_audio = src[s:e]
        seg_len = len(seg_audio)

        # If single segment exceeds hard max → split it
        if seg_len > hard_max_samples:

            split_pos = 0
            while split_pos < seg_len:
                end_pos = min(split_pos + hard_max_samples, seg_len)

                chunks.append({
                    "array": seg_audio[split_pos:end_pos].astype(np.float32),
                    "text": seg["text"]
                })

                split_pos = end_pos

            continue

        # If adding would exceed hard max → finalize current chunk first
        if current_samples + seg_len > hard_max_samples:
            if current_audio:
                chunks.append({
                    "array": np.concatenate(current_audio).astype(np.float32),
                    "text": " ".join(current_texts).strip()
                })

            current_audio = [seg_audio]
            current_texts = [seg["text"]]
            current_samples = seg_len
            continue

        # Otherwise add to current chunk
        current_audio.append(seg_audio)
        current_texts.append(seg["text"])
        current_samples += seg_len

        # If we crossed target length → finalize chunk
        if current_samples >= target_samples:
            chunks.append({
                "array": np.concatenate(current_audio).astype(np.float32),
                "text": " ".join(current_texts).strip()
            })

            current_audio = []
            current_texts = []
            current_samples = 0

    # Final tail (never drop)
    if current_audio:
        chunks.append({
            "array": np.concatenate(current_audio).astype(np.float32),
            "text": " ".join(current_texts).strip()
        })

    return chunks


# ──────────────────────────────────────────────
# Walk corpus
# ──────────────────────────────────────────────

def process_corpus(data_dir, chunk_sec, sr):

    all_chunks = []

    for root, _, files in os.walk(data_dir):

        file_set = set(files)

        audio_a_name = next((f for f in file_set
                             if f.startswith("speaker_a_convo") and f.endswith(".wav")), None)
        audio_b_name = next((f for f in file_set
                             if f.startswith("speaker_b_convo") and f.endswith(".wav")), None)
        json_a_name  = next((f for f in file_set
                             if f.startswith("speaker_a_convo") and f.endswith("_transcript.json")), None)
        json_b_name  = next((f for f in file_set
                             if f.startswith("speaker_b_convo") and f.endswith("_transcript.json")), None)

        if not all([audio_a_name, audio_b_name, json_a_name, json_b_name]):
            continue

        root_path = Path(root)
        log.info(f"Processing: {root_path}")

        try:
            audio_a = load_audio(root_path / audio_a_name, sr)
            audio_b = load_audio(root_path / audio_b_name, sr)

            segs_a = load_segments(root_path / json_a_name, "A")
            segs_b = load_segments(root_path / json_b_name, "B")

            chunks = build_chunks(
                audio_a,
                audio_b,
                segs_a,
                segs_b,
                target_chunk_sec=chunk_sec,
                hard_max_sec=40,
                sr=sr
            )

            log.info(f" → {len(chunks)} chunks")
            all_chunks.extend(chunks)

        except Exception as exc:
            log.warning(f" ✗ Skipped ({exc})")

    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


# ──────────────────────────────────────────────
# Build HF dataset
# ──────────────────────────────────────────────

def build_dataset(chunks, sr, val_frac, test_frac, seed):

    random.seed(seed)
    random.shuffle(chunks)

    n = len(chunks)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test

    def to_hf(data):
        ds = Dataset.from_dict({
            "audio": [{"array": c["array"], "sampling_rate": sr} for c in data],
            "text":  [c["text"] for c in data],
        })
        return ds.cast_column("audio", Audio(sampling_rate=sr))

    return DatasetDict({
        "train": to_hf(chunks[:n_train]),
        "validation": to_hf(chunks[n_train:n_train+n_val]),
        "test": to_hf(chunks[n_train+n_val:])
    })


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_sec", type=float, default=24.0)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--test_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    chunks = process_corpus(args.data_dir, args.chunk_sec, args.sr)

    dataset = build_dataset(
        chunks,
        args.sr,
        args.val_frac,
        args.test_frac,
        args.seed
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out))

    log.info(f"Saved dataset to {out}")
    print(dataset)


if __name__ == "__main__":
    main()