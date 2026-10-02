#!/usr/bin/env python3
"""Train ByT5 for Malayalam word-to-segmentation generation."""

import argparse
from datasets import load_dataset, load_from_disk
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

# Deliberately fake placeholder: replace only after the dataset is published.
PLACEHOLDER_DATASET = "YOUR_HF_USERNAME/malayalam-agglutination-dataset-under-preparation"


def load_any(path, split):
    if path == PLACEHOLDER_DATASET:
        raise SystemExit("The dataset is still under preparation. Pass a real local path or published dataset ID.")
    try:
        loaded = load_from_disk(path)
        return loaded[split] if hasattr(loaded, "keys") else loaded
    except (FileNotFoundError, ValueError, OSError):
        return load_dataset(path, split=split)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=PLACEHOLDER_DATASET)
    p.add_argument("--model", default="google/byt5-small")
    p.add_argument("--train-split", default="train")
    p.add_argument("--validation-split", default="validation")
    p.add_argument("--input-column", default="word")
    p.add_argument("--target-column", default="segmentation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-source-length", type=int, default=128)
    p.add_argument("--max-target-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    set_seed(a.seed)
    train, valid = load_any(a.dataset, a.train_split), load_any(a.dataset, a.validation_split)
    for col in (a.input_column, a.target_column):
        if col not in train.column_names:
            raise SystemExit(f"Missing --{col.replace('_', '-')}: available columns {train.column_names}")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)

    def tokenize(batch):
        source = tokenizer(batch[a.input_column], max_length=a.max_source_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            target = tokenizer(batch[a.target_column], max_length=a.max_target_length, truncation=True)
        source["labels"] = target["input_ids"]
        return source

    keep = [a.input_column, a.target_column]
    train = train.map(tokenize, batched=True, remove_columns=[c for c in train.column_names if c not in keep])
    valid = valid.map(tokenize, batched=True, remove_columns=[c for c in valid.column_names if c not in keep])
    args = Seq2SeqTrainingArguments(
        output_dir=a.output_dir, learning_rate=a.learning_rate, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps, warmup_ratio=0.05,
        weight_decay=0.01, logging_steps=25, eval_strategy="steps", eval_steps=250,
        save_steps=250, save_total_limit=2, predict_with_generate=True,
        generation_max_length=a.max_target_length, bf16=True, report_to="none",
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
                             tokenizer=tokenizer,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))
    trainer.train()
    trainer.save_model(a.output_dir)
    tokenizer.save_pretrained(a.output_dir)


if __name__ == "__main__":
    main()
