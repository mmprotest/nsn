"""Utilities for swapping PyTorch linear layers with :class:`NSNLinear`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from .init import initialise_nsn_from_linear
from .layers import NSNLinear

Selector = Callable[[str, nn.Module], bool]


@dataclass
class ReplacementRecord:
    name: str
    original_cls: str
    in_features: int
    out_features: int
    max_rank: int


def _default_selector(_: str, module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def replace_linear_with_nsn(
    model: nn.Module,
    selector: Optional[Selector] = None,
    *,
    max_rank: int,
    init: str = "svd",
    preserve_bias: bool = True,
    track_tied: bool = True,
) -> Tuple[nn.Module, List[ReplacementRecord]]:
    """Replace ``nn.Linear`` modules selected by ``selector`` with ``NSNLinear``.

    Parameters
    ----------
    model:
        The module whose submodules should be inspected.
    selector:
        Callable receiving ``(module_name, module)`` and returning ``True`` for
        modules that should be replaced. When ``None`` all linear layers are
        targeted.
    max_rank:
        The maximum NSN rank to use for each layer.
    init:
        Initialisation method name (currently only ``"svd"`` is supported).
    preserve_bias:
        Whether to keep and copy over linear biases.
    track_tied:
        If ``True`` weight-tied linears (same ``weight`` tensor) are only
        replaced once and subsequent occurrences are linked to the same
        :class:`NSNLinear` instance.
    """

    selector = selector or _default_selector
    replaced: List[ReplacementRecord] = []
    seen_weights: Dict[int, NSNLinear] = {}

    for module_name, module in list(model.named_modules())[::-1]:
        if not isinstance(module, nn.Linear):
            continue
        if not selector(module_name, module):
            continue

        parent, child_name = _locate_parent(model, module_name)
        if parent is None:
            continue

        tied_key = id(module.weight) if track_tied else None
        nsn_layer: Optional[NSNLinear] = None
        if tied_key is not None and tied_key in seen_weights:
            nsn_layer = seen_weights[tied_key]
        else:
            nsn_layer = initialise_nsn_from_linear(
                module, max_rank=max_rank, bias=preserve_bias
            )
            if tied_key is not None:
                seen_weights[tied_key] = nsn_layer

        setattr(parent, child_name, nsn_layer)
        replaced.append(
            ReplacementRecord(
                name=module_name,
                original_cls=module.__class__.__name__,
                in_features=module.in_features,
                out_features=module.out_features,
                max_rank=nsn_layer.max_rank,
            )
        )
    if not replaced:
        raise RuntimeError("Selector did not match any linear layer")
    return model, replaced


def _locate_parent(model: nn.Module, module_name: str) -> Tuple[Optional[nn.Module], str]:
    if module_name == "":
        return None, ""
    path = module_name.split(".")
    parent = model
    for name in path[:-1]:
        parent = getattr(parent, name)
    return parent, path[-1]


def restore_linear_from_nsn(
    model: nn.Module,
    *,
    fuse_rank: Optional[int] = None,
) -> nn.Module:
    """Restore dense :class:`nn.Linear` layers from :class:`NSNLinear` modules."""

    for module_name, module in list(model.named_modules())[::-1]:
        if not isinstance(module, NSNLinear):
            continue
        parent, child_name = _locate_parent(model, module_name)
        if parent is None:
            continue
        rank = fuse_rank or module.max_rank
        weight = module.effective_weight(rank)
        linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(weight)
            if module.bias is not None and linear.bias is not None:
                linear.bias.copy_(module.bias)
        setattr(parent, child_name, linear)
    return model


__all__ = ["replace_linear_with_nsn", "restore_linear_from_nsn", "ReplacementRecord"]
