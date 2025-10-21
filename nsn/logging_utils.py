"""Lightweight logging utilities for NSN experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rank_curve(csv_path: str | Path, out_path: str | Path, metric: str = "metric") -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    import pandas as pd  # type: ignore

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()
    ax.plot(df["rank"], df[metric], marker="o")
    ax.set_xlabel("Rank")
    ax.set_ylabel(metric)
    ax.grid(True)
    fig.savefig(out_path)
    plt.close(fig)


__all__ = ["append_jsonl", "write_csv", "plot_rank_curve"]
