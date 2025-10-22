# Nested Subspace Networks (NSN)

This repository contains a production-ready implementation of **Nested Subspace
Networks** for adapting existing PyTorch / Hugging Face `transformers` models.
NSN replaces selected dense `nn.Linear` layers with rank-controllable
`NSNLinear` layers that factorise the dense weight `W` into `B @ A`. At runtime
any rank `r \in [1, R]` can be activated to trade off accuracy against compute
and memory.

## Quickstart (60 seconds)

```bash
pip install -e .[dev]
python -m cli.nsn_adapt \
  --model EleutherAI/pythia-70m-deduped \
  --out ./pythia-nsn \
  --modules mlp \
  --max-rank 64
```

Now run generation at a reduced rank:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from nsn.utils import set_rank

tokenizer = AutoTokenizer.from_pretrained("./pythia-nsn")
model = AutoModelForCausalLM.from_pretrained("./pythia-nsn")
set_rank(model, 16)
print(model.generate(**tokenizer("Hello", return_tensors="pt")))
```

Alternatively, ``cli/nsn_infer.py`` can now wrap and run inference in a single
step. When pointed at a standard Hugging Face identifier it will adapt the
model automatically if required. GGUF checkpoints are also supported when a
tokenizer is provided:

```bash
python -m cli.nsn_infer \
  --model ./models/llama-gguf-q4_0.gguf \
  --tokenizer path/to/tokenizer \
  --prompt "Hello" \
  --target-flops 0.5
```

## Key Features

- **Post-hoc adaptation** via `nsn.wrap.replace_linear_with_nsn` or the
  high-level `nsn.adapt_hf_model` helper.
- **Rank control** with `nsn.utils.set_rank` / `get_rank` or
  `nsn.flops.select_rank_for_target_flops` for FLOPs-aware selection.
- **SVD initialisation** ensures rank-`R` behaviour matches the original
  dense layer out of the box.
- **Training utilities** including the uncertainty-weighted multi-rank loss,
  rank samplers, logging helpers, and checkpoint management.
- **Command line tools** for adapting (`cli/nsn_adapt.py`), training
  (`cli/nsn_train.py`), and inference (`cli/nsn_infer.py`).

## API Overview

```python
from nsn import (
    adapt_hf_model,
    estimate_model_flops,
    replace_linear_with_nsn,
    restore_linear_from_nsn,
    select_rank_for_target_flops,
    set_rank,
)
```

### Adapting a model

```python
from nsn import adapt_hf_model

model, tokenizer, records = adapt_hf_model("EleutherAI/pythia-70m-deduped", max_rank=64)
print(f"Adapted {len(records)} layers")
```

### Selecting ranks by FLOPs

```python
from nsn import select_rank_for_target_flops, set_rank

rank = select_rank_for_target_flops(model, target_fraction=0.5)
set_rank(model, rank)
```

### Restoring to dense

```python
from nsn import restore_linear_from_nsn

restore_linear_from_nsn(model)
```

## Examples

- `examples/hf_causal_lm_rank_sweep.py`: compute perplexity vs. rank and export
  a CSV ready for plotting.
- `examples/hf_seqcls_demo.py`: evaluate sequence classification accuracy at a
  chosen rank.

## Testing

```bash
pytest
```

All unit tests run on CPU and validate layer equivalence, SVD
initialisation quality, wrapping behaviour, uncertainty loss gradients, and
FLOPs estimates.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
