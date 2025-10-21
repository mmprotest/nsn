from __future__ import annotations

import argparse

from nsn.flops import select_rank_for_target_flops
from nsn.utils import set_rank

try:  # transformers optional
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = AutoTokenizer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with NSN ranks")
    parser.add_argument("--model", required=True, help="Model directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--target-flops", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required for inference")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.target_flops is not None:
        rank = select_rank_for_target_flops(model, args.target_flops)
        set_rank(model, rank)
    elif args.rank is not None:
        set_rank(model, args.rank)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
