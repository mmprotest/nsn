"""Generic helpers shared across the package."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from .layers import NSNLinear


def iter_nsn_modules(module: nn.Module) -> Iterable[NSNLinear]:
    for submodule in module.modules():
        if isinstance(submodule, NSNLinear):
            yield submodule


def set_rank(module: nn.Module, rank: int) -> None:
    """Set the active rank of every :class:`NSNLinear` within ``module``."""

    for nsn in iter_nsn_modules(module):
        nsn.set_rank(rank)


def get_rank(module: nn.Module) -> Optional[int]:
    """Return the shared rank of all :class:`NSNLinear` modules if equal."""

    current: Optional[int] = None
    for nsn in iter_nsn_modules(module):
        if current is None:
            current = nsn.get_rank()
        elif current != nsn.get_rank():
            return None
    return current


def ensure_same_dtype(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != other.dtype:
        return tensor.to(dtype=other.dtype)
    return tensor


__all__ = ["set_rank", "get_rank", "iter_nsn_modules", "ensure_same_dtype"]
