"""Loss functions for training NSN layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch import Tensor


@dataclass
class UncertaintyLossOutput:
    loss: Tensor
    per_rank_losses: Dict[int, Tensor]
    uncertainties: Dict[int, Tensor]


class RankUncertaintyLoss(torch.nn.Module):
    """Jointly optimises multiple ranks with uncertainty weighting."""

    def __init__(self, ranks: Sequence[int]):
        super().__init__()
        self.ranks = list(sorted(set(ranks)))
        self.log_sigmas = torch.nn.Parameter(torch.zeros(len(self.ranks)))

    def forward(self, losses: Sequence[Tensor]) -> UncertaintyLossOutput:  # type: ignore[override]
        if len(losses) != len(self.ranks):
            raise ValueError("Losses sequence must align with configured ranks")
        log_sigmas = self.log_sigmas
        per_rank = {}
        weighted_terms = []
        for idx, (rank, loss) in enumerate(zip(self.ranks, losses)):
            sigma = log_sigmas[idx]
            weight = torch.exp(-sigma)
            term = weight * loss + sigma
            per_rank[rank] = term.detach()
            weighted_terms.append(term)
        total = torch.stack(weighted_terms).sum()
        return UncertaintyLossOutput(
            loss=total,
            per_rank_losses={r: l.detach() for r, l in zip(self.ranks, losses)},
            uncertainties={r: s for r, s in zip(self.ranks, log_sigmas)},
        )


__all__ = ["RankUncertaintyLoss", "UncertaintyLossOutput"]
