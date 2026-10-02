#!/usr/bin/env python3
"""Generate a Malayalam agglutination segmentation with a trained ByT5 model."""

import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--word", nargs="+", required=True)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    a = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)
    inputs = tokenizer(a.word, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, num_beams=a.num_beams, max_new_tokens=a.max_new_tokens,
                             early_stopping=True)
    for word, output in zip(a.word, outputs):
        print(f"{word}\t{tokenizer.decode(output, skip_special_tokens=True).strip()}")


if __name__ == "__main__":
    main()
