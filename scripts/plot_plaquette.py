#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plaquette_stats import (  # noqa: E402
    extract_plaquette,
    fmt_mean_err,
    mean_jackknife,
    parse_traj_range,
)


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
    parser.add_argument(
        "--binsize",
        type=int,
        default=100,
        help="Jackknife block/bin size (default: 100)",
    )
    args = parser.parse_args()

    space_vals, time_vals, mean_vals = extract_plaquette(args.logfile)
    if not mean_vals:
        raise SystemExit("No plaquette lines found in log.")

    total = len(mean_vals)
    traj_min, traj_max = parse_traj_range(args.traj, total)
    if traj_min > total:
        raise SystemExit(
            f"Requested --traj starts at {traj_min}, but log only has "
            f"{total} plaquette point(s); nothing to plot."
        )

    start = traj_min - 1
    end = traj_max
    space_vals = space_vals[start:end]
    time_vals = time_vals[start:end]
    mean_vals = mean_vals[start:end]
    if not mean_vals:
        raise SystemExit("Selected trajectory window is empty.")
    x = list(range(traj_min, traj_max + 1))
    n_sel = len(mean_vals)

    space_mean, space_sem, n_blocks, n_used = mean_jackknife(
        space_vals, binsize=args.binsize
    )
    time_mean, time_sem, _, _ = mean_jackknife(time_vals, binsize=args.binsize)
    st_mean, st_sem, _, _ = mean_jackknife(mean_vals, binsize=args.binsize)
    s_str = fmt_mean_err(space_mean, space_sem)
    t_str = fmt_mean_err(time_mean, time_sem)
    m_str = fmt_mean_err(st_mean, st_sem)
    val_width = max(len(s_str), len(t_str), len(m_str))

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
        f"Jackknife binsize={args.binsize}, blocks={n_blocks}, used={n_used}\n"
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
    print(
        f"Jackknife: binsize={args.binsize}, blocks={n_blocks}, "
        f"used={n_used} (dropped {n_sel - n_used})."
    )
    print(f"Spatial  : {s_str:>{val_width}}")
    print(f"Temporal : {t_str:>{val_width}}")
    print(f"Mean(ST) : {m_str:>{val_width}}")
    print(f"Saved plot to: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
