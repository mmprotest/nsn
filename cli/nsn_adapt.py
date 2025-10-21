from __future__ import annotations

import argparse
from pathlib import Path

from nsn.hf_integration import adapt_hf_model
from nsn.checkpoint import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt a HF model with NSN layers")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--task", default="causal-lm", help="Task type")
    parser.add_argument("--max-rank", type=int, default=64, help="Maximum NSN rank")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=["mlp"],
        help="Module presets to adapt (e.g. mlp attn)",
    )
    parser.add_argument("--init", default="svd", help="Initialisation method")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, records = adapt_hf_model(
        args.model,
        task=args.task,
        max_rank=args.max_rank,
        modules=args.modules,
        init=args.init,
        trust_remote_code=args.trust_remote_code,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out_dir / "nsn.ckpt")
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    try:
        model.save_pretrained(out_dir)
    except AttributeError:
        pass
    print(f"Adapted {len(records)} linear layers. Saved to {out_dir}.")


if __name__ == "__main__":
    main()
