"""Rank sampling utilities."""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence


def uniform_sampler(ranks: Sequence[int], k: int) -> List[int]:
    """Sample ``k`` ranks uniformly without replacement."""

    ranks = list(sorted(set(ranks)))
    if k >= len(ranks):
        return list(ranks)
    return random.sample(ranks, k)


def low_bias_sampler(ranks: Sequence[int], k: int) -> List[int]:
    """Sampler biased towards smaller ranks."""

    ranks = list(sorted(set(ranks)))
    weights = [1 / (idx + 1) for idx in range(len(ranks))]
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen = []
    while len(chosen) < min(k, len(ranks)):
        choice = random.choices(ranks, weights=probs, k=1)[0]
        if choice not in chosen:
            chosen.append(choice)
    return chosen


def curriculum_sampler(ranks: Sequence[int], k: int, step: int) -> List[int]:
    """Gradually increase the maximum sampled rank as ``step`` grows."""

    ranks = list(sorted(set(ranks)))
    max_index = min(len(ranks), step + 1)
    return uniform_sampler(ranks[:max_index], k)


__all__ = ["uniform_sampler", "low_bias_sampler", "curriculum_sampler"]
