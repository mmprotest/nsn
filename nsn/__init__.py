"""Top-level package for Nested Subspace Networks (NSN).

This module exposes the main public APIs for adapting Hugging Face
``transformers`` models with :class:`~nsn.layers.NSNLinear` layers,
working with FLOPs estimations, and controlling active ranks globally.
"""

from __future__ import annotations

from .flops import estimate_model_flops, select_rank_for_target_flops
from .hf_integration import adapt_hf_model
from .layers import NSNLinear
from .utils import get_rank, set_rank
from .wrap import replace_linear_with_nsn, restore_linear_from_nsn

__all__ = [
    "NSNLinear",
    "adapt_hf_model",
    "estimate_model_flops",
    "select_rank_for_target_flops",
    "replace_linear_with_nsn",
    "restore_linear_from_nsn",
    "set_rank",
    "get_rank",
]
