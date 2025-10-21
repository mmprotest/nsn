"""Core NSN layer implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class NSNState:
    """Small dataclass for storing cached weight metadata."""

    weight: Optional[Tensor]
    a_version: int
    b_version: int


class NSNLinear(nn.Module):
    """A low-rank controllable linear layer using the NSN parameterisation.

    The layer factorises the dense weight ``W`` into ``B @ A`` where
    ``A`` has shape ``(max_rank, in_features)`` and ``B`` has shape
    ``(out_features, max_rank)``. At inference time the caller can choose
    an active rank ``r`` (``1 <= r <= max_rank``) to trade accuracy for
    compute.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int,
        *,
        bias: bool = True,
        init_method: str = "svd",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        min_rank: int = 1,
    ) -> None:
        super().__init__()
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if max_rank > min(in_features, out_features):
            raise ValueError(
                "max_rank must be <= min(in_features, out_features)"
            )
        if min_rank < 1 or min_rank > max_rank:
            raise ValueError("min_rank must satisfy 1 <= min_rank <= max_rank")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.max_rank_value = int(max_rank)
        self.min_rank_value = int(min_rank)
        self.init_method = init_method

        self.A = nn.Parameter(torch.empty(max_rank, in_features, **factory_kwargs))
        self.B = nn.Parameter(torch.empty(out_features, max_rank, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.active_rank = max_rank
        self._cache: Optional[NSNState] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Simple orthogonal initialisation used when we cannot rely on SVD.
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
        self._invalidate_cache()

    @property
    def max_rank(self) -> int:  # type: ignore[override]
        return self.max_rank_value

    @property
    def min_rank(self) -> int:
        return self.min_rank_value

    def set_rank(self, rank: int) -> None:
        if not (self.min_rank_value <= rank <= self.max_rank_value):
            raise ValueError(
                f"rank must be within [{self.min_rank_value}, {self.max_rank_value}]"
            )
        self.active_rank = int(rank)

    def get_rank(self) -> int:
        return int(self.active_rank)

    def forward(self, x: Tensor, rank: Optional[int] = None) -> Tensor:  # type: ignore[override]
        if rank is None:
            rank = self.active_rank
        if not (self.min_rank_value <= rank <= self.max_rank_value):
            raise ValueError(
                f"rank must be within [{self.min_rank_value}, {self.max_rank_value}]"
            )

        if rank == self.max_rank_value:
            weight = self._get_cached_weight()
            y = torch.nn.functional.linear(x, weight, self.bias)
            return y

        a_r = self.A[:rank, :]
        b_r = self.B[:, :rank]
        # Compute using two matrix multiplications for numerical efficiency.
        x_shape = x.shape
        x_mat = x.reshape(-1, self.in_features)
        tmp = torch.matmul(x_mat, a_r.t())
        out = torch.matmul(tmp, b_r.t())
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x_shape[:-1], self.out_features)

    def _invalidate_cache(self) -> None:
        self._cache = None

    def _get_cached_weight(self) -> Tensor:
        a_version = self.A._version
        b_version = self.B._version
        if self._cache is None or (
            self._cache.a_version != a_version or self._cache.b_version != b_version
        ):
            weight = torch.matmul(self.B, self.A)
            self._cache = NSNState(weight=weight, a_version=a_version, b_version=b_version)
        assert self._cache.weight is not None
        return self._cache.weight

    def effective_weight(self, rank: Optional[int] = None) -> Tensor:
        if rank is None:
            rank = self.active_rank
        if rank == self.max_rank_value:
            return self._get_cached_weight()
        a_r = self.A[:rank, :]
        b_r = self.B[:, :rank]
        return torch.matmul(b_r, a_r)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"max_rank={self.max_rank_value}, active_rank={self.active_rank}"
        )


__all__ = ["NSNLinear"]
