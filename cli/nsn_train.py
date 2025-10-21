from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader

from nsn.loss import RankUncertaintyLoss
from nsn.rank_sampling import uniform_sampler
from nsn.utils import set_rank

try:  # optional dependencies
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    load_dataset = AutoModelForCausalLM = AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an NSN-adapted model")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--data", required=True, help="Dataset name for datasets.load_dataset")
    parser.add_argument("--task", default="causal-lm")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--r-min", type=int, default=4)
    parser.add_argument("--r-sample-per-step", type=int, default=2)
    parser.add_argument("--loss-uncertainty", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def _prepare_dataloader(tokenizer, dataset_name: str, batch_size: int) -> DataLoader:
    dataset = load_dataset(dataset_name, split="train")

    def collate(batch: List[dict]) -> dict:
        texts = [item[next(iter(item.keys()))] for item in batch]
        encoded = tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": encoded["attention_mask"]}

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)


def main() -> None:
    args = parse_args()
    if AutoModelForCausalLM is None or load_dataset is None:
        raise ImportError("transformers and datasets are required for training")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    dataloader = _prepare_dataloader(tokenizer, args.data, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    nsn_layers = [m for m in model.modules() if hasattr(m, "max_rank") and hasattr(m, "set_rank")]
    max_rank = max(getattr(layer, "max_rank") for layer in nsn_layers)
    ranks = list(range(args.r_min, max_rank + 1))
    rank_loss = RankUncertaintyLoss([max_rank] + ranks[: args.r_sample_per_step - 1])

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            sampled = uniform_sampler(ranks, args.r_sample_per_step)
            if max_rank not in sampled:
                sampled.append(max_rank)
            losses = []
            for rank in sampled:
                set_rank(model, rank)
                outputs = model(**batch)
                losses.append(outputs.loss)
            loss_out = rank_loss(losses)
            optimiser.zero_grad()
            loss_out.loss.backward()
            optimiser.step()
            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={loss_out.loss.item():.4f}")
        set_rank(model, max_rank)
    model.save_pretrained(args.model)


if __name__ == "__main__":
    main()
