from __future__ import annotations

import torch
from torch import nn

from nsn.init import initialise_nsn_from_linear
from nsn.layers import NSNLinear


def test_forward_matches_linear():
    torch.manual_seed(0)
    linear = nn.Linear(8, 4, bias=True)
    nsn = initialise_nsn_from_linear(linear, max_rank=4)
    x = torch.randn(2, 8)
    linear_out = linear(x)
    nsn_out = nsn(x)
    assert torch.allclose(linear_out, nsn_out, atol=1e-5)


def test_rank_switching():
    torch.manual_seed(0)
    linear = nn.Linear(16, 16)
    nsn = initialise_nsn_from_linear(linear, max_rank=8)
    x = torch.randn(3, 16)
    full = nsn(x, rank=8)
    nsn.set_rank(4)
    reduced = nsn(x)
    assert full.shape == reduced.shape
    assert not torch.allclose(full, reduced)


@torch.autocast("cpu", enabled=True, dtype=torch.bfloat16)
def test_autocast_forward():
    linear = nn.Linear(4, 4)
    nsn = initialise_nsn_from_linear(linear, max_rank=4)
    x = torch.randn(2, 4)
    out = nsn(x)
    assert out.dtype == torch.bfloat16
