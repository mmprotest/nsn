from __future__ import annotations

import torch
from torch import nn

from nsn.init import initialise_nsn_from_linear
from nsn.layers import NSNLinear
from nsn.utils import set_rank


class ToyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(6, 6)
        self.layer2 = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(x)))


def _adapt_model(model: nn.Module, max_rank: int = 6) -> nn.Module:
    for name, module in list(model.named_modules())[::-1]:
        if isinstance(module, nn.Linear):
            nsn = initialise_nsn_from_linear(module, max_rank=max_rank)
            parent = model
            if name:
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], nsn)
    return model


def test_rank_metric_curve():
    torch.manual_seed(0)
    model = ToyRegressor()
    adapted = _adapt_model(model, max_rank=6)
    x = torch.randn(64, 6)
    y = torch.sin(x.sum(dim=-1, keepdim=True))
    max_rank = min(layer.max_rank for layer in adapted.modules() if hasattr(layer, "max_rank"))
    ranks = [1, max(1, max_rank // 2), max_rank]
    losses = []
    for rank in ranks:
        set_rank(adapted, rank)
        pred = adapted(x)
        losses.append(torch.nn.functional.mse_loss(pred, y).item())
    assert losses[0] >= losses[-1] - 1e-6
