"""Configuration dataclasses used throughout the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NSNConfig:
    max_rank: int = 64
    min_rank: int = 1
    modules: List[str] = field(default_factory=lambda: ["mlp"])
    init: str = "svd"


@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    epochs: int = 1
    gradient_clip: Optional[float] = None
    bf16: bool = False
    fp16: bool = False
    rank_samples_per_step: int = 2
    log_every: int = 10


__all__ = ["NSNConfig", "TrainConfig"]
