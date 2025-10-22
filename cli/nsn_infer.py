from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nsn.checkpoint import save_checkpoint
from nsn.flops import select_rank_for_target_flops
from nsn.gguf_integration import load_gguf_causal_lm
from nsn.hf_integration import adapt_hf_model
from nsn.utils import is_nsn_model, set_rank

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
    parser.add_argument("--task", default="causal-lm", help="Task type for adaptation")
    parser.add_argument("--max-rank", type=int, default=64, help="Maximum NSN rank")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=["mlp"],
        help="Module presets to adapt when wrapping",
    )
    parser.add_argument("--init", default="svd", help="Initialisation method")
    parser.add_argument("--tokenizer", help="Tokenizer path or name (required for GGUF)")
    parser.add_argument("--save-processed", help="Optional directory to save the adapted model")
    parser.add_argument("--device", help="Device for inference (e.g. cuda)")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype when loading GGUF weights",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = None
    model = None
    dtype = getattr(torch, args.dtype)
    model_path = Path(args.model)

    if model_path.suffix.lower() == ".gguf":
        model, tokenizer = load_gguf_causal_lm(model_path, tokenizer_path=args.tokenizer, dtype=dtype)
    else:
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required for inference")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        tokenizer_source = args.tokenizer or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=args.trust_remote_code
        )

    if tokenizer is None:
        raise ValueError("A tokenizer is required for generation; provide --tokenizer for GGUF models")

    adapted = False
    if not is_nsn_model(model):
        modules = args.modules or ["mlp"]
        model, _, _ = adapt_hf_model(
            model,
            task=args.task,
            max_rank=args.max_rank,
            modules=modules,
            init=args.init,
            trust_remote_code=args.trust_remote_code,
        )
        adapted = True

    if adapted and args.save_processed:
        out_dir = Path(args.save_processed)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, out_dir / "nsn.ckpt")
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    device = args.device
    if device is not None:
        model.to(device)
    else:
        device = next(model.parameters()).device

    if args.target_flops is not None:
        rank = select_rank_for_target_flops(model, args.target_flops)
        set_rank(model, rank)
    elif args.rank is not None:
        set_rank(model, args.rank)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
