#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PATTERN = re.compile(
    r"Plaquette:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"\(\s*spatial:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"temporal:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)"
)


def extract_plaquette(log_path: Path):
    space_vals = []
    time_vals = []
    mean_vals = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            mean_vals.append(float(m.group(1)))
            space_vals.append(float(m.group(2)))
            time_vals.append(float(m.group(3)))

    return space_vals, time_vals, mean_vals


def parse_traj_range(raw: str, total: int):
    m = re.fullmatch(r"\s*(\d*)\s*:\s*(\d*)\s*", raw)
    if not m:
        raise SystemExit("Invalid --traj format. Use MIN:MAX, :MAX or MIN:")

    left = m.group(1)
    right = m.group(2)
    if left == "" and right == "":
        raise SystemExit("Invalid --traj: both MIN and MAX are empty.")

    traj_min = 1 if left == "" else max(1, int(left))
    traj_max = total if right == "" else min(total, int(right))
    if traj_min > traj_max:
        raise SystemExit(
            f"Invalid range: traj-min ({traj_min}) > traj-max ({traj_max})."
        )
    return traj_min, traj_max


def mean_sem(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size < 2:
        sem = 0.0
    else:
        sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, sem


def fmt_mean_err(mean, sem):
    mean_str = f"{mean:.6f}"
    err_digits = int(round(abs(sem) * 1_000_000.0))
    return f"{mean_str}({err_digits})"


def main():
    parser = argparse.ArgumentParser(
        description="Extract plaquette values from run log and plot them."
    )
    parser.add_argument("logfile", type=Path, help="Path to terminal/log text file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plaquette_plot.png"),
        help="Output image path (default: plaquette_plot.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window in addition to saving figure",
    )
    parser.add_argument(
        "--traj",
        type=str,
        default=None,
        help="Trajectory range MIN:MAX (1-based, inclusive), e.g. 5:20",
    )
    args = parser.parse_args()

    space_vals, time_vals, mean_vals = extract_plaquette(args.logfile)
    if not mean_vals:
        raise SystemExit("No plaquette lines found in log.")

    total = len(mean_vals)
    if args.traj is None:
        traj_min, traj_max = 1, total
    else:
        traj_min, traj_max = parse_traj_range(args.traj, total)

    start = traj_min - 1
    end = traj_max
    space_vals = space_vals[start:end]
    time_vals = time_vals[start:end]
    mean_vals = mean_vals[start:end]
    x = list(range(traj_min, traj_max + 1))
    n_sel = len(mean_vals)

    space_mean, space_sem = mean_sem(space_vals)
    time_mean, time_sem = mean_sem(time_vals)
    st_mean, st_sem = mean_sem(mean_vals)
    s_str = fmt_mean_err(space_mean, space_sem)
    t_str = fmt_mean_err(time_mean, time_sem)
    m_str = fmt_mean_err(st_mean, st_sem)
    val_width = max(len(s_str), len(t_str), len(m_str))

    # Journal-like plotting style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "lines.markersize": 2.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(x, space_vals, color="#1f77b4", label="Spatial")
    ax.plot(x, time_vals, color="#d62728", label="Temporal")
    ax.plot(
        x,
        mean_vals,
        color="#111111",
        linestyle="-.",
        linewidth=2.0,
        label="Mean (space-time)",
    )
    ax.set_xlabel("Trajectory index")
    ax.set_ylabel("Plaquette")
    ax.set_title("Plaquette Evolution")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.20)
    ax.legend(loc="best", frameon=False)
    stats_text = (
        f"Selected traj: [{traj_min}, {traj_max}], N={n_sel}\n"
        f"Spatial   = {s_str:>{val_width}}\n"
        f"Temporal  = {t_str:>{val_width}}\n"
        f"Mean(ST)  = {m_str:>{val_width}}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.5"),
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")

    print(f"Extracted {total} plaquette points in total.")
    print(f"Plotted trajectory range: [{traj_min}, {traj_max}] ({n_sel} points).")
    print(f"Spatial  : {s_str:>{val_width}}")
    print(f"Temporal : {t_str:>{val_width}}")
    print(f"Mean(ST) : {m_str:>{val_width}}")
    print(f"Saved plot to: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
