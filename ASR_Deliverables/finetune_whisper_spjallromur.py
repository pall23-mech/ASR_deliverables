# ===============================================================
# Whisper-Large Icelandic — Fine-tuning on Spjallrómur (local)
# ===============================================================

import re
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, List, Dict

from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate

# ===============================================================
# Reproducibility
# ===============================================================
SEED = 42
set_seed(SEED)

# ===============================================================
# Config  ← only things you should ever need to change
# ===============================================================
MODEL_ID    = "language-and-voice-lab/whisper-large-icelandic-62640-steps-967h"
DATASET_DIR = "./spjallromur_hf_dataset"   # <-- set path to your local dataset here
OUTPUT_DIR  = "whisper-large-icelandic-spjall"
TEXT_COLUMN = "text"                        # column name in your local dataset

# ── Subset flag ──────────────────────────────────────────────
# True  → use a small fixed number of examples per split (fast smoke-test)
# False → use the full dataset
SUBSET   = False
SUBSET_N = 10    # examples per split when SUBSET=True

# ── Hardware ─────────────────────────────────────────────────
USE_CPU = False   # True → force CPU; False → use GPU if available

# ===============================================================
# Model + processor
# ===============================================================
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="icelandic", task="transcribe")
processor.tokenizer.set_prefix_tokens(language="icelandic", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.language  = "icelandic"
model.config.task      = "transcribe"
model.config.use_cache = False

# Freeze encoder — only fine-tune the decoder.
# With a small local dataset this prevents overfitting the encoder.
for param in model.model.encoder.parameters():
    param.requires_grad = False

# Load or build generation config, then patch it
try:
    gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
except Exception:
    gen_cfg = GenerationConfig.from_model_config(model.config)

forced_decoder_ids = processor.get_decoder_prompt_ids(language="icelandic", task="transcribe")
gen_cfg.forced_decoder_ids = forced_decoder_ids
gen_cfg.max_new_tokens     = 444

# Empty list causes a HF warning; None is the correct "disabled" value
if isinstance(gen_cfg.suppress_tokens, list) and len(gen_cfg.suppress_tokens) == 0:
    gen_cfg.suppress_tokens = None

model.generation_config = gen_cfg

# ===============================================================
# Dataset
# ===============================================================
dataset = load_from_disk(DATASET_DIR)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

train_dataset = dataset["train"]
eval_dataset  = dataset["validation"]
test_dataset  = dataset["test"]

print("Split sizes:", {
    "train":      len(train_dataset),
    "validation": len(eval_dataset),
    "test":       len(test_dataset),
})

if SUBSET:
    train_dataset = train_dataset.select(range(min(SUBSET_N, len(train_dataset))))
    eval_dataset  = eval_dataset.select(range(min(SUBSET_N, len(eval_dataset))))
    test_dataset  = test_dataset.select(range(min(SUBSET_N, len(test_dataset))))
    print(f"Subset active ({SUBSET_N} examples per split):", {
        "train":      len(train_dataset),
        "validation": len(eval_dataset),
        "test":       len(test_dataset),
    })

# ===============================================================
# Text normalisation (WER scoring only — labels are kept raw)
# ===============================================================
def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\sáéíóúýðþæö]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# ===============================================================
# Feature extraction
# ===============================================================
def preprocess(batch: Dict) -> Dict:
    """Convert raw audio + text into model inputs. Works on batches of any size."""
    input_features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for audio, text in zip(batch["audio"], batch[TEXT_COLUMN]):
        encoded = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=text,
            return_tensors="pt",
        )
        input_features_list.append(encoded.input_features[0])
        labels_list.append(encoded.labels[0])

    return {"input_features": input_features_list, "labels": labels_list}


def get_remove_cols(ds):
    return [c for c in ds.column_names if c not in ("input_features", "labels")]


train_dataset = train_dataset.map(
    preprocess, batched=True, batch_size=32,
    remove_columns=get_remove_cols(train_dataset),
    desc="Preprocessing train",
)
eval_dataset = eval_dataset.map(
    preprocess, batched=True, batch_size=32,
    remove_columns=get_remove_cols(eval_dataset),
    desc="Preprocessing validation",
)
test_dataset = test_dataset.map(
    preprocess, batched=True, batch_size=32,
    remove_columns=get_remove_cols(test_dataset),
    desc="Preprocessing test",
)

# ===============================================================
# Data collator
# ===============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]}              for f in features]

        batch        = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features,         return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if labels.size(1) > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

# ===============================================================
# Metrics
# ===============================================================
metric = evaluate.load("wer")

def compute_metrics(pred) -> Dict[str, float]:
    pred_ids  = np.array(pred.predictions)
    label_ids = np.array(pred.label_ids)

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids,           skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids.tolist(), skip_special_tokens=True)

    wer_value = metric.compute(
        predictions=[normalize_text(p) for p in pred_str],
        references =[normalize_text(l) for l in label_str],
    )
    return {"wer": round(100 * wer_value, 2)}


# ===============================================================
# Training
# ===============================================================
if __name__ == "__main__":

    use_gpu = not USE_CPU and torch.cuda.is_available()
    print(f"Training on: {'GPU' if use_gpu else 'CPU'}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch / gradient
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,

        # Learning rate
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        num_train_epochs=10,
        weight_decay=0.01,

        # Precision
        fp16=use_gpu,                    # fp16 only on GPU
        no_cuda=USE_CPU,                 # force CPU when USE_CPU=True

        # Evaluation & checkpointing
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        # Local-friendly settings
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,        # 0 required on Windows; safe everywhere
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5,
                                         early_stopping_threshold=0.5)],
    )

    trainer.train()

    # ── Held-out test evaluation ────────────────────────────────
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    # ── Save ────────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model and processor saved to '{OUTPUT_DIR}'")