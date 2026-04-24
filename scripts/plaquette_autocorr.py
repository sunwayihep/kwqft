#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PATTERN = re.compile(
    r"Plaquette:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
    r"\(\s*spatial:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"temporal:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)"
)


def extract_series(log_path: Path, component: str) -> np.ndarray:
    values = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            vals = {"mean": float(m.group(1)), "space": float(m.group(2)), "time": float(m.group(3))}
            values.append(vals[component])
    return np.asarray(values, dtype=float)


def parse_traj(raw: str | None, total: int) -> tuple[int, int]:
    if raw is None:
        return 1, total
    m = re.fullmatch(r"\s*(\d*)\s*:\s*(\d*)\s*", raw)
    if not m:
        raise SystemExit("Invalid --traj format. Use MIN:MAX, :MAX or MIN:")
    left, right = m.group(1), m.group(2)
    if left == "" and right == "":
        raise SystemExit("Invalid --traj: both MIN and MAX are empty.")
    lo = 1 if left == "" else max(1, int(left))
    hi = total if right == "" else min(total, int(right))
    if lo > hi:
        raise SystemExit(f"Invalid range: traj-min ({lo}) > traj-max ({hi}).")
    return lo, hi


def rho_t(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    if n < 2:
        raise ValueError("Need at least two points for autocorrelation.")
    var = np.var(x)
    if var == 0.0:
        return np.ones(n)
    rho = np.empty(n, dtype=float)
    for t in range(n):
        rho[t] = np.dot(x[: n - t], x[t:]) / (n - t)
    rho /= rho[0]
    return rho


def tau_int_curve(rho: np.ndarray) -> np.ndarray:
    # tau_int(W) = 0.5 + sum_{t=1..W} rho(t)
    n = rho.size
    tau = np.empty(n, dtype=float)
    tau[0] = 0.5
    csum = 0.0
    for w in range(1, n):
        csum += rho[w]
        tau[w] = 0.5 + csum
    return tau


def choose_window(tau: np.ndarray, c: float = 5.0) -> int:
    # Simple Madras-Sokal-like self-consistent window
    w_star = 1
    for w in range(1, tau.size):
        if w >= c * tau[w]:
            w_star = w
            break
        w_star = w
    return w_star


def main():
    p = argparse.ArgumentParser(
        description="Compute plaquette autocorrelation and integrated autocorrelation time."
    )
    p.add_argument("logfile", type=Path, help="Path to run log file")
    p.add_argument(
        "--component",
        choices=["space", "time", "mean"],
        default="mean",
        help="Which plaquette component to analyze (default: mean)",
    )
    p.add_argument(
        "--traj",
        type=str,
        default=None,
        help="Trajectory range MIN:MAX (1-based, inclusive), e.g. 10:200",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("plaquette_autocorr.png"),
        help="Output figure path",
    )
    p.add_argument("--show", action="store_true", help="Show figure window")
    args = p.parse_args()

    series = extract_series(args.logfile, args.component)
    if series.size < 4:
        raise SystemExit("Not enough plaquette points found (need >= 4).")

    lo, hi = parse_traj(args.traj, series.size)
    y = series[lo - 1 : hi]
    n = y.size
    if n < 4:
        raise SystemExit("Selected trajectory range is too short (need >= 4).")

    rho = rho_t(y)
    tau = tau_int_curve(rho)
    w_star = choose_window(tau, c=5.0)
    tau_star = tau[w_star]
    stride_2tau = max(1, math.ceil(2.0 * tau_star))
    stride_5tau = max(1, math.ceil(5.0 * tau_star))

    lags = np.arange(n)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=False)

    ax1.plot(lags, rho, color="#1f77b4", lw=1.8)
    ax1.axhline(0.0, color="black", lw=0.8, alpha=0.7)
    ax1.set_xlabel("Traj.")
    ax1.set_ylabel(r"$\rho(t)$")
    ax1.set_title(f"Plaquette autocorrelation $\\rho(t)$ ({args.component}), traj [{lo}, {hi}]")
    ax1.grid(True, ls="--", lw=0.6, alpha=0.35)

    w = np.arange(n)
    ax2.plot(w[1:], tau[1:], color="#d62728", lw=1.8, label=r"$\tau_{\mathrm{int}}(W)$")
    ax2.axvline(w_star, color="#111111", ls="--", lw=1.0, label=f"W*={w_star}")
    ax2.set_xlabel("Window W")
    ax2.set_ylabel(r"$\tau_{\mathrm{int}}$")
    ax2.grid(True, ls="--", lw=0.6, alpha=0.35)
    ax2.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")

    print(f"Extracted {series.size} points total; used range [{lo}, {hi}] -> N={n}")
    print(f"Estimated tau_int = {tau_star:.6g} at W*={w_star} (c=5 criterion)")
    print(
        "Suggested save stride (in trajectories): "
        f">= {stride_2tau} (~2*tau_int, light decorrelation), "
        f">= {stride_5tau} (~5*tau_int, conservative)."
    )
    print(f"Saved figure to: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
