"""Checkpoint helpers for NSN layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
from torch import nn

from .layers import NSNLinear
from .utils import iter_nsn_modules


@dataclass
class LayerCheckpoint:
    name: str
    in_features: int
    out_features: int
    max_rank: int


def _gather_named_nsn(module: nn.Module) -> Iterator[Tuple[str, NSNLinear]]:
    for name, submodule in module.named_modules():
        if isinstance(submodule, NSNLinear):
            yield name, submodule


def save_checkpoint(module: nn.Module, path: str | Path) -> None:
    path = Path(path)
    payload: Dict[str, Dict[str, torch.Tensor]] = {}
    metadata: Dict[str, Dict[str, int]] = {}
    for name, nsn in _gather_named_nsn(module):
        payload[name] = {
            "A": nsn.A.detach().cpu(),
            "B": nsn.B.detach().cpu(),
            "bias": nsn.bias.detach().cpu() if nsn.bias is not None else None,  # type: ignore[assignment]
        }
        metadata[name] = {
            "in_features": nsn.in_features,
            "out_features": nsn.out_features,
            "max_rank": nsn.max_rank,
        }
    torch.save({"parameters": payload, "metadata": metadata}, path)


def load_checkpoint(module: nn.Module, path: str | Path) -> None:
    data = torch.load(path, map_location="cpu")
    params = data["parameters"]
    for name, nsn in _gather_named_nsn(module):
        if name not in params:
            continue
        param_data = params[name]
        nsn.A.data.copy_(param_data["A"].to(nsn.A.device))
        nsn.B.data.copy_(param_data["B"].to(nsn.B.device))
        if nsn.bias is not None and param_data["bias"] is not None:
            nsn.bias.data.copy_(param_data["bias"].to(nsn.bias.device))


def merge_to_dense(module: nn.Module) -> nn.Module:
    from .wrap import restore_linear_from_nsn

    return restore_linear_from_nsn(module)


__all__ = ["save_checkpoint", "load_checkpoint", "merge_to_dense", "LayerCheckpoint"]
