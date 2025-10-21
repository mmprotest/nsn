from __future__ import annotations

import torch
from torch import nn

from nsn.flops import estimate_model_flops, flops_linear, flops_nsn, select_rank_for_target_flops
from nsn.init import initialise_nsn_from_linear
from nsn.utils import set_rank


class Simple(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(10, 12)
        self.l2 = nn.Linear(12, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


def test_flops_estimates():
    torch.manual_seed(0)
    model = Simple()
    for name, module in list(model.named_modules())[::-1]:
        if isinstance(module, nn.Linear):
            parent = model
            if name:
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], initialise_nsn_from_linear(module, max_rank=6))
    full = estimate_model_flops(model, 6)
    reduced = estimate_model_flops(model, 3)
    assert reduced < full
    rank = select_rank_for_target_flops(model, 0.5)
    assert model.l1.max_rank >= rank >= 1
