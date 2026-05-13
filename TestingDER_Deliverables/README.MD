# DER Evaluation Harness — Spjallrómur Conversational ASR for Icelandic

**Author:** Páll Rúnarsson, Language and Voice Laboratory, Reykjavík University  
**Contact:** pallr@ru.is  
**License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)  
**Last updated:** March 2026

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey) ![Last updated: Mar 2026](https://img.shields.io/badge/last%20updated-Mar%202026-green)

General-purpose Diarization Error Rate (DER) evaluation harness for any diarization backend against any HuggingFace speech dataset with gold speaker annotations. Supports local inference, pre-computed RTTM files, and cloud APIs, with resumable runs, failure isolation, and diagnostic plots.

---

## Background

Accurate speaker diarization is crucial for downstream ASR tasks like speaker-attributed transcription, meeting summarization, and conversational analytics — especially in low-resource languages like Icelandic. This harness enables fair, offline comparison of local models versus cloud APIs versus external baselines on the same dataset, with consistent evaluation methodology throughout.

**Expected DER range:** pyannote 3.1 typically achieves 20–35% DER on Icelandic conversational data like Spjallrómur without domain-specific fine-tuning. Results vary with recording conditions, speaker overlap, and noise level.

---

## Supported backends

| `--backend`      | System                                                        | Needs GPU?   |
|------------------|---------------------------------------------------------------|--------------|
| `pyannote_local` | pyannote/speaker-diarization-3.1 (local inference)            | Recommended  |
| `rttm`           | Any external system — supply pre-computed RTTM files          | No           |
| `aws`            | AWS Transcribe (speaker diarization job)                      | No           |
| `revai`          | Rev.ai (speaker diarization job)                              | No           |

---

## Project structure

```
TestingDER_Deliverables/
├── der_eval/
│   ├── __init__.py      # package version
│   ├── config.py        # CONFIG defaults + argparse
│   ├── audio.py         # materialise dataset audio to WAV files
│   ├── backends.py      # backend init + per-backend diarize()
│   ├── metrics.py       # reference annotation + evaluation loop
│   ├── report.py        # console summary + scatter plot
│   └── __main__.py      # wires all modules together
├── der_eval.py          # thin top-level entry point
└── requirements.txt
```

---

## Installation

### 1. Install system dependencies

ffmpeg is required by pyannote for audio decoding:

```bash
# Ubuntu / Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Conda (any platform)
conda install -c conda-forge ffmpeg
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install PyTorch

PyTorch must be installed before the rest of the requirements:

```bash
# CUDA 12.1  (recommended — RTX 30xx / 40xx, driver >= 525)
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8  (older GPUs / locked drivers)
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch==2.3.1 torchaudio==2.3.1
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify

```bash
python -c "import torch, torchaudio, pyannote.audio; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('pyannote OK')"
```

---

## HuggingFace authentication

The `pyannote/speaker-diarization-3.1` model is gated. You must:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Log in:

```bash
huggingface-cli login
# paste your hf_... token when prompted
```

Alternatively, pass the token directly at runtime:

```bash
python der_eval.py --hf-token hf_yourtoken
```

---

## Configuration

All defaults live in `der_eval/config.py` under the `DEFAULTS` dict. Edit that block to permanently change any setting. Every key can also be overridden at runtime via CLI arguments.

| Key             | Default                              | Description                              |
|-----------------|--------------------------------------|------------------------------------------|
| `backend`       | `pyannote_local`                     | Which diarization system to use          |
| `dataset_id`    | `palli23/Spjallromur-AB-Diarization-test` | HuggingFace dataset                 |
| `split`         | `test`                               | Dataset split, supports slice syntax     |
| `audio_col`     | `audio`                              | Name of the audio column                 |
| `segments_col`  | `segments`                           | Name of the gold segments column         |
| `duration_col`  | `duration`                           | Name of the duration column              |
| `audio_out_dir` | `diar_eval_audio`                    | Where materialised WAVs are written      |
| `results_csv`   | `diarization_eval_results.csv`       | Per-file DER results                     |
| `plot_png`      | `der_vs_duration.png`                | Output scatter plot                      |

---

## Usage

### Basic run

```bash
python der_eval.py
```

### Quick smoke test (2 files)

```bash
python der_eval.py --split "test[:2]"
```

### Switch backend

```bash
# Pre-computed RTTM files
python der_eval.py --backend rttm --rttm-dir ./my_rttm_files

# AWS Transcribe
python der_eval.py --backend aws

# Rev.ai
python der_eval.py --backend revai
```

### Custom dataset or output paths

```bash
python der_eval.py --dataset my_org/my_dataset --split "test"
python der_eval.py --results my_results.csv --plot my_plot.png
```

### All CLI options

```
--backend     {pyannote_local,rttm,aws,revai}
--dataset     HuggingFace dataset ID
--split       Dataset split, e.g. "test" or "test[:50]"
--hf-token    HuggingFace access token
--rttm-dir    Directory containing pre-computed RTTM files
--audio-dir   Directory where WAV files are written
--results     Output CSV path
--plot        Output PNG path
```

You can also run the package directly:

```bash
python -m der_eval --split "test[:2]"
```

---

## RTTM backend

If you have diarization output from an external system, export it to RTTM format and use the `rttm` backend. Place all files in a single directory named with a zero-padded index prefix matching the dataset row:

```
rttm_output/
    0000_<anything>.rttm
    0001_<anything>.rttm
    ...
```

Each file must follow the standard 10-field RTTM format:

```
SPEAKER <file_id> 1 <start_sec> <duration_sec> <NA> <NA> <speaker_id> <NA> <NA>
```

Example:

```
SPEAKER conv_001 1 0.000 4.500 <NA> <NA> spk_00 <NA> <NA>
SPEAKER conv_001 1 4.800 3.200 <NA> <NA> spk_01 <NA> <NA>
```

---

## Output

### Console summary

```
=============================================
  Diarization Evaluation Summary
=============================================
  Backend   : pyannote_local
  Dataset   : palli23/Spjallromur-AB-Diarization-test  (test)
  Rows processed : 10  |  DER computed: 10
  Unweighted average DER : 0.3502  (35.02%)
  Duration-weighted DER  : 0.3491  (34.91%)
  Std deviation          : 0.1014  (10.14%)
  Results CSV : diarization_eval_results.csv
=============================================
```

### CSV columns

| Column        | Description                              |
|---------------|------------------------------------------|
| `index`       | Dataset row index                        |
| `backend`     | Backend used                             |
| `dataset`     | HuggingFace dataset ID                   |
| `split`       | Split used                               |
| `audio_path`  | Path to materialised WAV                 |
| `duration`    | Clip duration in seconds                 |
| `num_segments`| Number of gold speaker segments          |
| `der`         | DER as a fraction (0–1+)                 |
| `der_percent` | DER as a percentage                      |

Failed rows are written to a separate `*_failures.csv` file.

### Plot

A DER-vs-duration scatter plot is saved to `der_vs_duration.png`.

---

## Cloud API setup

### AWS Transcribe

Set credentials in `der_eval/config.py` or as environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

The IAM user needs `transcribe:StartTranscriptionJob`, `transcribe:GetTranscriptionJob`, and `s3:PutObject` permissions.

### Rev.ai

```bash
export REVAI_ACCESS_TOKEN=...
```

Or set `revai_access_token` in `der_eval/config.py`.

---

## License

DER Evaluation Harness © 2026 by Páll Rúnarsson is licensed under
[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to use, share, and adapt this work for any purpose, including commercially, provided you give appropriate credit and distribute any derivatives under the same license.

---

## Acknowledgements

Developed at the Language and Voice Laboratory, Reykjavík University, as part of the nationally funded Almannarómur project *Towards Automatic Speech Recognition for Conversations in Icelandic* (MVF25010163). Research conducted under the supervision of Professor Jón Guðnason. Thanks to Michal Borsky for valuable technical discussions and review throughout the project. Corpus provided via CLARIN-IS / Spjallrómur.
