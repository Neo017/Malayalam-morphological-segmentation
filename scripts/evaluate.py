#!/usr/bin/env python3
"""Evaluate exact segmentation and morpheme-component F1."""

import argparse
import json
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PLACEHOLDER_DATASET = "YOUR_HF_USERNAME/malayalam-agglutination-dataset-under-preparation"


def load_any(path, split):
    if path == PLACEHOLDER_DATASET:
        raise SystemExit("The dataset is under preparation; pass a real evaluation dataset.")
    try:
        loaded = load_from_disk(path)
        return loaded[split] if hasattr(loaded, "keys") else loaded
    except (FileNotFoundError, ValueError, OSError):
        return load_dataset(path, split=split)


def components(segmentation):
    return {p.strip() for p in segmentation.split("+") if p.strip()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--input-column", default="word")
    p.add_argument("--target-column", default="segmentation")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-beams", type=int, default=4)
    a = p.parse_args()
    data = load_any(a.dataset, a.split)
    tok, model = AutoTokenizer.from_pretrained(a.model), AutoModelForSeq2SeqLM.from_pretrained(a.model)
    exact = tp = fp = fn = invalid = 0
    for start in range(0, len(data), a.batch_size):
        rows = data[start:start + a.batch_size]
        words, gold = rows[a.input_column], rows[a.target_column]
        inputs = tok(words, return_tensors="pt", padding=True, truncation=True)
        generated = model.generate(**inputs, num_beams=a.num_beams, max_new_tokens=256)
        pred = [tok.decode(x, skip_special_tokens=True).strip() for x in generated]
        for word, ref, hyp in zip(words, gold, pred):
            exact += hyp == ref
            if "+" not in hyp:
                invalid += 1
            g, h = components(ref), components(hyp)
            tp += len(g & h); fp += len(h - g); fn += len(g - h)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(json.dumps({"examples": len(data), "exact_match": exact / len(data),
                      "component_precision": precision, "component_recall": recall,
                      "component_f1": f1, "invalid_output_rate": invalid / len(data)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
