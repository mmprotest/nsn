from __future__ import annotations

import torch

from nsn.loss import RankUncertaintyLoss


def test_uncertainty_loss_gradient_flow():
    torch.manual_seed(0)
    criterion = RankUncertaintyLoss([2, 4])
    losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
    out = criterion(losses)
    out.loss.backward()
    for loss in losses:
        assert loss.grad is not None and loss.grad.item() != 0
    assert criterion.log_sigmas.grad is not None
