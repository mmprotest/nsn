"""Utilities for initialising NSN layers from dense linear layers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .layers import NSNLinear


def truncated_svd(
    weight: Tensor,
    rank: int,
    *,
    deterministic: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute a truncated SVD of ``weight`` returning ``U, S, Vh``.

    The function first attempts to call :func:`torch.linalg.svd`. When this is
    not possible (e.g. because the matrix is very large and running on CPU), a
    lightweight power-iteration based approximation is used instead. The helper
    is intentionally self-contained to avoid a hard dependency on SciPy.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    out_features, in_features = weight.shape
    max_rank = min(out_features, in_features)
    if rank > max_rank:
        raise ValueError("rank cannot exceed min(out_features, in_features)")

    try:
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        return U[:, :rank], S[:rank], Vh[:rank, :]
    except RuntimeError:
        pass

    # Fallback: randomised range finder via power iteration.
    device = weight.device
    dtype = weight.dtype
    random = torch.randn(in_features, rank, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(weight.t() @ random)
    for _ in range(2 if deterministic else 4):
        q, _ = torch.linalg.qr(weight.t() @ (weight @ q))
    b = weight @ q
    u_tilde, s, vh_tilde = torch.linalg.svd(b, full_matrices=False)
    U = u_tilde[:, :rank]
    S = s[:rank]
    Vh = (vh_tilde[:rank, :] @ q.t())
    return U, S, Vh


def initialise_nsn_from_linear(
    linear: nn.Linear,
    *,
    max_rank: int,
    nsn_layer: Optional[NSNLinear] = None,
    bias: bool = True,
) -> NSNLinear:
    """Create or populate an :class:`NSNLinear` from a dense ``nn.Linear``."""

    weight = linear.weight.detach()
    device = weight.device
    dtype = weight.dtype
    out_features, in_features = weight.shape
    max_rank = min(max_rank, min(out_features, in_features))

    if nsn_layer is None:
        nsn_layer = NSNLinear(
            in_features,
            out_features,
            max_rank,
            bias=bias and linear.bias is not None,
            device=device,
            dtype=dtype,
        )
    else:
        if nsn_layer.in_features != in_features or nsn_layer.out_features != out_features:
            raise ValueError("nsn_layer dimensions do not match the linear layer")

    with torch.no_grad():
        U, S, Vh = truncated_svd(weight, max_rank)
        sqrt_s = torch.sqrt(S)
        B = U * sqrt_s.unsqueeze(0)
        A = (sqrt_s.unsqueeze(1) * Vh)
        nsn_layer.B.copy_(B)
        nsn_layer.A.copy_(A)
        if nsn_layer.bias is not None and linear.bias is not None:
            nsn_layer.bias.copy_(linear.bias)
        elif nsn_layer.bias is not None:
            nsn_layer.bias.zero_()
    nsn_layer.set_rank(max_rank)
    return nsn_layer


__all__ = ["truncated_svd", "initialise_nsn_from_linear"]
