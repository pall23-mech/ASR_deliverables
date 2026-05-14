# Spjallrómur— Conversational ASR for Icelandic

**Author:** Páll Rúnarsson, Reykjavík University / Language and Voice Lab  
**Contact:** pallr@ru.is  
**License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by/4.0/)  
**Last updated:** March 2026

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey) ![Last updated: Mar 2026](https://img.shields.io/badge/last%20updated-Mar%202026-green)

This repository contains the full research and evaluation pipeline developed as part of the **Almannarómur** máltækniáætlun nationally funded speech processing project Towards Automatic Speech Recognition for Conversations in Icelandic. The work focuses on advancing Icelandic automatic speech recognition for conversational domains using the Spjallrómur corpus.

---
> **Note:** The `aws` and `revai` backends are experimental. Cloud API behaviour, 
> quota limits, and audio format requirements vary by account and region. 
> `pyannote_local` and `rttm` are the recommended backends for reproducible results.


## Repository structure

```
project/
├── ASR_Deliverables/           # Whisper fine-tuning recipe + dataset builder
├── TestingWER_Deliverables/    # Word Error Rate evaluation harness
└── TestingDER_Deliverables/    # Diarization Error Rate evaluation harness
```

---

## Components

### `ASR_Deliverables/` — Whisper Fine-Tuning

Contains the dataset builder for the Spjallrómur corpus (CLARIN-IS) and a full Whisper-Large fine-tuning script. Start here if you want to train or reproduce the conversational ASR model.

See `ASR_Deliverables/README.md` for setup and usage instructions.

**Key result:** Fine-tuning on Spjallrómur reduces WER from 46.2% → 28.6% over read-speech adaptation alone (~38% relative improvement).

---

### `TestingWER_Deliverables/` — WER Evaluation

Offline, backend-agnostic Word Error Rate evaluation harness. Supports any HuggingFace-compatible ASR model, fine-tuned Whisper checkpoints, or pre-computed hypotheses via JSON files. Outputs per-sample WER/CER, S/D/I breakdown, JSON + CSV results, and a scatter plot.

See `TestingWER_Deliverables/README.md` for setup and usage instructions.

---

### `TestingDER_Deliverables/` — DER Evaluation

Offline Diarization Error Rate evaluation harness supporting pyannote (local), pre-computed RTTM files, AWS Transcribe, and Rev.ai. Outputs per-file DER, CSV results, and a scatter plot.

See `TestingDER_Deliverables/README.md` for setup and usage instructions.

**Expected DER range:** pyannote 3.1 achieves 20–35% DER on Spjallrómur without domain-specific fine-tuning.

---

## Background

Accurate ASR and speaker diarization for Icelandic conversational speech remains an open challenge due to the low-resource nature of the language and the significant domain gap between read and spontaneous speech. This project addresses both problems through targeted fine-tuning and reproducible, backend-agnostic evaluation tooling that enables fair comparison across systems.

---

## License

Spjallrómur Evaluation Suite © 2026 by Páll Rúnarsson is licensed under CC BY-SA 4.0
  
Creative Commons Attribution-ShareAlike 4.0 International
This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, even for commercial purposes. If others remix, adapt, or build upon the material, they must license the modified material under identical terms.(https://creativecommons.org/licenses/by/4.0/)

---

## Acknowledgements

Developed at Reykjavík University / Language and Voice Lab in collaboration with Tiro as part of the nationally funded Almannarómur project. Research conducted with advisory input from Professor Jón Guðnason (Language and Voice Lab, Reykjavík University) and Dr. Michal Borsky (Tiro). Corpus provided via CLARIN-IS / Spjallrómur. Base Whisper model (whisper-large-icelandic-62640-steps-967h) developed by Carlos Hernandez at the Language and Voice Lab.
