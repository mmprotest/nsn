"""Top-level package for Nested Subspace Networks (NSN).

This module exposes the main public APIs for adapting Hugging Face
``transformers`` models with :class:`~nsn.layers.NSNLinear` layers,
working with FLOPs estimations, and controlling active ranks globally.
"""

from __future__ import annotations

from .flops import estimate_model_flops, select_rank_for_target_flops
from .gguf_integration import load_gguf_causal_lm
from .hf_integration import adapt_hf_model
from .layers import NSNLinear
from .utils import get_rank, is_nsn_model, set_rank
from .wrap import replace_linear_with_nsn, restore_linear_from_nsn

__all__ = [
    "NSNLinear",
    "adapt_hf_model",
    "load_gguf_causal_lm",
    "estimate_model_flops",
    "select_rank_for_target_flops",
    "replace_linear_with_nsn",
    "restore_linear_from_nsn",
    "set_rank",
    "get_rank",
    "is_nsn_model",
]
