#!/usr/bin/env python3
"""Shared plaquette log parsing and blocked-jackknife statistics.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PATTERN = re.compile(
    r"Plaquette:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"\(\s*spatial:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"temporal:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)"
)


def extract_plaquette(log_path: Path):
    """Return (spatial, temporal, mean) plaquette series from a log file."""
    space_vals: list[float] = []
    time_vals: list[float] = []
    mean_vals: list[float] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            mean_vals.append(float(m.group(1)))
            space_vals.append(float(m.group(2)))
            time_vals.append(float(m.group(3)))
    return space_vals, time_vals, mean_vals


def parse_traj_range(raw: str | None, total: int) -> tuple[int, int]:
    """Parse MIN:MAX trajectory window (1-based, inclusive). None → full range.

    If ``MIN`` is beyond ``total`` (run still incomplete), returns
    ``(MIN, total)`` with ``MIN > total`` so callers can skip that dataset
    instead of aborting.
    """
    if raw is None:
        return 1, total
    m = re.fullmatch(r"\s*(\d*)\s*:\s*(\d*)\s*", raw)
    if not m:
        raise SystemExit("Invalid --traj format. Use MIN:MAX, :MAX or MIN:")
    left, right = m.group(1), m.group(2)
    if left == "" and right == "":
        raise SystemExit("Invalid --traj: both MIN and MAX are empty.")
    traj_min = 1 if left == "" else max(1, int(left))
    if right == "":
        traj_max = total
    else:
        traj_max = min(total, int(right))

    # Incomplete run: requested start is past available measurements.
    if traj_min > total:
        return traj_min, total

    if traj_min > traj_max:
        raise SystemExit(
            f"Invalid range: traj-min ({traj_min}) > traj-max ({traj_max})."
        )
    return traj_min, traj_max


def mean_jackknife(values, binsize: int = 100):
    """Mean and blocked delete-1 jackknife error.

    Data are partitioned into contiguous blocks of length ``binsize``.
    Trailing points that do not fill a full block are dropped. The reported
    mean uses the retained samples; the uncertainty is the jackknife SEM
    over leave-one-block-out means.

    Returns
    -------
    mean, err, n_blocks, n_used
    """
    arr = np.asarray(values, dtype=float)
    if binsize < 1:
        raise SystemExit(f"Invalid --binsize ({binsize}): must be >= 1")
    if arr.size == 0:
        return 0.0, 0.0, 0, 0

    n_blocks = arr.size // binsize
    if n_blocks == 0:
        mean = float(np.mean(arr))
        if arr.size < 2:
            return mean, 0.0, 0, int(arr.size)
        sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        return mean, sem, 0, int(arr.size)

    used = arr[: n_blocks * binsize]
    mean = float(np.mean(used))
    if n_blocks == 1:
        return mean, 0.0, 1, int(used.size)

    blocks = used.reshape(n_blocks, binsize)
    block_sums = blocks.sum(axis=1)
    total_sum = float(block_sums.sum())
    n_used = float(used.size)
    jk = (total_sum - block_sums) / (n_used - binsize)
    jk_mean = float(np.mean(jk))
    err = float(
        np.sqrt((n_blocks - 1) / n_blocks * np.sum((jk - jk_mean) ** 2))
    )
    return mean, err, n_blocks, int(used.size)


def fmt_mean_err(mean: float, sem: float) -> str:
    mean_str = f"{mean:.8f}"
    err_digits = int(round(abs(sem) * 100_000_000.0))
    return f"{mean_str}({err_digits})"
