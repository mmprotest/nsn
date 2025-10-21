from __future__ import annotations

import torch
from torch import nn

from nsn.wrap import replace_linear_with_nsn, restore_linear_from_nsn


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.linear2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


def test_replace_and_restore_roundtrip():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.randn(2, 8)
    baseline = model(x)
    model, records = replace_linear_with_nsn(model, max_rank=8)
    assert records
    adapted = model(x)
    restored = restore_linear_from_nsn(model)
    after = restored(x)
    assert torch.allclose(adapted, after, atol=1e-5)
    assert torch.allclose(baseline, after, atol=1e-3)
