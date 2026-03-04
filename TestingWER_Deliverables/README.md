# WER Evaluation Harness — Málrómur / Spjallrómur

**Author:** Páll Rúnarsson, Reykjavík University  
**Contact:** pallr@ru.is  
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
**Last updated:** March 2026

Offline, backend-agnostic Word Error Rate evaluation for Icelandic ASR.
Outputs per-sample JSON + CSV + a WER-vs-duration scatter plot.

---

## Backends

| `--backend`   | What it runs                                            |
|---------------|---------------------------------------------------------|
| `whisper`     | Any `openai/whisper-*` or fine-tuned Whisper checkpoint |
| `hf_pipeline` | Any HF `automatic-speech-recognition` model (wav2vec2, MMS, …) |
| `json`        | Pre-computed hypotheses from a directory of JSON files  |

---

## Quick start

```bash
# 1. Install PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install everything else
pip install -r requirements.txt

# 3. Run — Whisper small, first 100 samples
python wer_eval.py

# 4. Run — Whisper large-v3
python wer_eval.py --model-id openai/whisper-large-v3

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
--backend        whisper | hf_pipeline | json      (default: whisper)
--model-id       HF model ID or local path         (default: openai/whisper-small)
--dataset        HF dataset ID
--split          e.g. "train[:100]"
--audio-col      audio column name                 (default: audio)
--text-col       raw reference column              (default: text)
--norm-text-col  normalised reference column       (default: norm_text)
--language       ISO-639-1 language code           (default: is)
--max-new-tokens                                   (default: 225)
--batch-size                                       (default: 8)
--json-dir       directory of hypothesis JSON files
--results-json   output JSON path
--results-csv    output CSV path
--plot           output PNG path
--n-examples     worst samples printed in summary  (default: 10)
--hf-token       HuggingFace access token
--device         cuda | cpu  (default: auto)
```

---

## Output files

| File                    | Contents                                         |
|-------------------------|--------------------------------------------------|
| `wer_results.json`      | Summary + per-sample ref / hyp / WER / CER / RTF |
| `wer_results.csv`       | Same, spreadsheet-friendly                        |
| `wer_vs_duration.png`   | Scatter plot with linear trend + median line      |

---

## Acknowledgements

Dataset, processing code, and Arrow format by Páll Rúnarsson, Reykjavík University.  
Developed as part of the nationally funded Málrómur speech processing project.
