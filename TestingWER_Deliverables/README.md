# WER Evaluation Harness — Spjallrómur Conversational ASR for Icelandic

**Author:** Páll Rúnarsson, Language and Voice Laboratory, Reykjavík University  
**Contact:** pallr@ru.is  
**License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)  
**Last updated:** March 2026

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License: CC BY-SA 4.0](https://img.shields.io/badge/license-CC%20BY--SA%204.0-lightgrey) ![Last updated: Mar 2026](https://img.shields.io/badge/last%20updated-Mar%202026-green)

Offline, backend-agnostic Word Error Rate (WER) evaluation harness for Icelandic ASR. Supports any HuggingFace-compatible ASR model, fine-tuned Whisper checkpoints, or pre-computed hypotheses via JSON files. Outputs per-sample WER/CER/RTF, S/D/I breakdown, JSON + CSV results, and a WER-vs-duration scatter plot.

---

## Background

Robust ASR evaluation requires consistent normalisation, reproducible tooling, and the ability to compare across model families without rewriting evaluation code for each backend. This harness provides a single evaluation interface for Whisper, wav2vec2, MMS, and any other HuggingFace-compatible ASR model, as well as pre-computed hypotheses from external systems — enabling fair, apples-to-apples comparison on the same dataset and split.

---

## Supported backends

| `--backend`   | What it runs                                                    |
|---------------|-----------------------------------------------------------------|
| `whisper`     | Any `openai/whisper-*` or fine-tuned Whisper checkpoint         |
| `hf_pipeline` | Any HF `automatic-speech-recognition` model (wav2vec2, MMS, …) |
| `json`        | Pre-computed hypotheses from a directory of JSON files          |

---

## Quick start

```bash
# 1. Install PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install everything else
pip install -r requirements.txt

# 3. Run — Whisper small, first 100 samples (smoke test)
python wer_eval.py --split "test[:100]"

# 4. Run — fine-tuned Whisper large
python wer_eval.py --model-id language-and-voice-lab/whisper-large-icelandic-62640-steps-967h

# 5. Run — any other HF model
python wer_eval.py --backend hf_pipeline \
    --model-id facebook/mms-300m

# 6. Run — pre-computed hypotheses (no GPU needed)
python wer_eval.py --backend json --json-dir ./my_transcripts/
```

---

## JSON backend format

One file per sample, zero-padded filename matching the dataset index:

```
my_transcripts/
  00000.json
  00001.json
  ...
```

Each file:

```json
{
  "hypothesis": "þetta er dæmi um málsgrein",
  "reference":  "þetta er dæmi um málsgrein"
}
```

`"reference"` is optional — if absent, the dataset's reference column is used.

---

## All options

```
--backend        whisper | hf_pipeline | json          (default: whisper)
--model-id       HF model ID or local path             (default: openai/whisper-small)
--dataset        HF dataset ID                         (default: palli23/spjallromur-4h)
--split          e.g. "test" or "test[:100]"           (default: test)
--audio-col      audio column name                     (default: audio)
--text-col       raw reference column                  (default: text)
--norm-text-col  normalised reference column           (default: norm_text)
--language       ISO-639-1 language code               (default: is)
--max-new-tokens                                       (default: 225)
--batch-size                                           (default: 8)
--json-dir       directory of hypothesis JSON files
--results-json   output JSON path                      (default: wer_results.json)
--results-csv    output CSV path                       (default: wer_results.csv)
--plot           output PNG path                       (default: wer_vs_duration.png)
--n-examples     worst samples printed in summary      (default: 10)
--hf-token       HuggingFace access token
--device         cuda | cpu                            (default: auto)
```

---

## Output

### Console summary

```
=============================================
  WER Evaluation Summary
=============================================
  Backend   : whisper
  Model     : language-and-voice-lab/whisper-large-icelandic-62640-steps-967h
  Dataset   : palli23/spjallromur-4h  (test)
  Samples processed : 100
  Overall WER       : 0.2860  (28.60%)
  Overall CER       : 0.1340  (13.40%)
  Median RTF        : 0.21
  Results JSON : wer_results.json
  Results CSV  : wer_results.csv
=============================================

10 worst samples by WER:
  [idx=42]  WER=0.81  ref: "hvað er eiginlega að gerast hér"
                       hyp: "hvað eiginlega gerast"
  ...
```

### Output files

| File                  | Contents                                              |
|-----------------------|-------------------------------------------------------|
| `wer_results.json`    | Summary + per-sample ref / hyp / WER / CER / RTF      |
| `wer_results.csv`     | Same, spreadsheet-friendly                            |
| `wer_vs_duration.png` | Scatter plot with linear trend + median line          |

### CSV columns

| Column       | Description                            |
|--------------|----------------------------------------|
| `index`      | Dataset row index                      |
| `backend`    | Backend used                           |
| `model_id`   | Model identifier                       |
| `dataset`    | HuggingFace dataset ID                 |
| `split`      | Split used                             |
| `duration`   | Clip duration in seconds               |
| `reference`  | Normalised reference text              |
| `hypothesis` | Model output text                      |
| `wer`        | WER as a fraction                      |
| `cer`        | CER as a fraction                      |
| `rtf`        | Real-time factor                       |

---

## Reproducibility

- WER normalisation is applied consistently to both references and hypotheses
- Default split is `test` — do not use `train` or `validation` for final reported results
- Random seed is not required; evaluation is deterministic given fixed model weights
- Dependency versions are pinned in `requirements.txt`

---

## License

WER Evaluation Harness © 2026 by Páll Rúnarsson is licensed under
[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

You are free to use, share, and adapt this work for any purpose, including commercially, provided you give appropriate credit and distribute any derivatives under the same license.

---

## Acknowledgements

Developed at the Language and Voice Laboratory, Reykjavík University, as part of the nationally funded Almannarómur project *Towards Automatic Speech Recognition for Conversations in Icelandic* (MVF25010163). Research conducted under the supervision of Professor Jón Guðnason. Thanks to Michal Borsky for valuable technical discussions and review throughout the project. Corpus provided via CLARIN-IS / Spjallrómur.
