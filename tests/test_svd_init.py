from __future__ import annotations

import torch
from torch import nn

from nsn.init import initialise_nsn_from_linear


def test_svd_reconstruction_error():
    torch.manual_seed(0)
    linear = nn.Linear(12, 10)
    nsn = initialise_nsn_from_linear(linear, max_rank=10)
    weight_dense = linear.weight
    weight_nsn = nsn.effective_weight(10)
    error = torch.norm(weight_dense - weight_nsn) / torch.norm(weight_dense)
    assert error < 1e-5


def test_rank_monotonic_improvement():
    torch.manual_seed(0)
    linear = nn.Linear(20, 16)
    nsn = initialise_nsn_from_linear(linear, max_rank=8)
    low = nsn.effective_weight(2)
    mid = nsn.effective_weight(4)
    full = nsn.effective_weight(8)
    target = linear.weight
    def rel_err(mat):
        return torch.norm(target - mat) / torch.norm(target)
    assert rel_err(mid) <= rel_err(low) + 1e-6
    assert rel_err(full) <= rel_err(mid) + 1e-6
