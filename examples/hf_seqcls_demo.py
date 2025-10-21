"""Sequence classification demo using NSN ranks."""

from __future__ import annotations

import argparse

import torch

from nsn.utils import set_rank

try:  # optional dependencies
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover
    load_dataset = AutoModelForSequenceClassification = AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence classification demo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if AutoModelForSequenceClassification is None or load_dataset is None:
        raise ImportError("transformers and datasets are required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    if args.rank is not None:
        set_rank(model, args.rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = load_dataset("glue", args.dataset, split=args.split)
    correct = 0
    total = 0
    for sample in dataset.select(range(64)):
        encoded = tokenizer(sample["sentence"], return_tensors="pt").to(device)
        outputs = model(**encoded)
        pred = outputs.logits.argmax(dim=-1).item()
        correct += int(pred == sample["label"])
        total += 1
    acc = correct / max(total, 1)
    print(f"Accuracy over {total} samples at rank {args.rank or 'default'}: {acc:.3f}")


if __name__ == "__main__":
    main()
