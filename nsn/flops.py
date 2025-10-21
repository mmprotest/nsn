"""FLOPs accounting for NSN layers."""

from __future__ import annotations

from typing import Iterable, Tuple

from torch import nn

from .layers import NSNLinear
from .utils import iter_nsn_modules


def flops_linear(in_features: int, out_features: int) -> int:
    return 2 * in_features * out_features


def flops_nsn(in_features: int, out_features: int, rank: int) -> int:
    return 2 * rank * (in_features + out_features)


def _collect_layers(module: nn.Module) -> Tuple[list[NSNLinear], int]:
    nsn_layers = list(iter_nsn_modules(module))
    other_flops = 0
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear) and not isinstance(submodule, NSNLinear):
            other_flops += flops_linear(submodule.in_features, submodule.out_features)
    return nsn_layers, other_flops


def estimate_model_flops(module: nn.Module, rank: int) -> int:
    nsn_layers, other_flops = _collect_layers(module)
    total = other_flops
    for nsn in nsn_layers:
        total += flops_nsn(nsn.in_features, nsn.out_features, rank)
    return total


def select_rank_for_target_flops(module: nn.Module, target_fraction: float) -> int:
    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must be within (0, 1]")
    nsn_layers, other_flops = _collect_layers(module)
    if not nsn_layers:
        raise RuntimeError("Model does not contain any NSN layers")

    max_rank = max(layer.max_rank for layer in nsn_layers)
    min_rank = min(layer.min_rank for layer in nsn_layers)

    max_total = other_flops
    for nsn in nsn_layers:
        max_total += flops_nsn(nsn.in_features, nsn.out_features, nsn.max_rank)

    target = max_total * target_fraction

    low, high = min_rank, max_rank
    best_rank = high
    while low <= high:
        mid = (low + high) // 2
        current = other_flops
        for nsn in nsn_layers:
            current += flops_nsn(nsn.in_features, nsn.out_features, min(mid, nsn.max_rank))
        if current <= target:
            best_rank = mid
            high = mid - 1
        else:
            low = mid + 1
    return max(min_rank, min(best_rank, max_rank))


__all__ = ["estimate_model_flops", "select_rank_for_target_flops", "flops_linear", "flops_nsn"]
