"""Example script that evaluates perplexity across NSN ranks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from nsn.flops import estimate_model_flops
from nsn.logging_utils import write_csv
from nsn.utils import set_rank

try:  # optional dependencies
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    load_dataset = AutoModelForCausalLM = AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank sweep for perplexity")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--ranks", nargs="*", type=int)
    parser.add_argument("--out", default="rank_sweep.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if AutoModelForCausalLM is None or load_dataset is None:
        raise ImportError("transformers and datasets are required for this example")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = load_dataset(args.dataset, split=args.split)
    ranks = args.ranks or list(range(4, max(layer.max_rank for layer in model.modules() if hasattr(layer, "max_rank")) + 1, 4))
    rows = []
    for rank in ranks:
        set_rank(model, rank)
        losses = []
        for sample in dataset.select(range(8)):
            text = sample[next(iter(sample.keys()))]
            encoded = tokenizer(text, return_tensors="pt").to(device)
            labels = encoded["input_ids"].clone()
            outputs = model(**encoded, labels=labels)
            losses.append(outputs.loss.detach())
        ppl = math.exp(torch.stack(losses).mean().item())
        flops_fraction = estimate_model_flops(model, rank) / estimate_model_flops(
            model, max(layer.max_rank for layer in model.modules() if hasattr(layer, "max_rank"))
        )
        rows.append({"rank": rank, "flops_fraction": flops_fraction, "perplexity": ppl})
        print(f"rank={rank} ppl={ppl:.2f} flops_fraction={flops_fraction:.3f}")
    write_csv(args.out, rows)


if __name__ == "__main__":
    main()
