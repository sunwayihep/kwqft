#!/usr/bin/env python3
"""
Check unitarity and plaquette values of KWQFT binary gauge configurations.

Binary layout (no header): site-major order, each site stores NDIMS SU(N) link
matrices as complex128 in C row-major e[i][j] order.

The lattice parameters are parsed from the KWQFT heatbath filename, e.g.:
  su3_nd4_beta6_L8_L8_L8_L16_cfg_50.bin

Example:
  python test_plaquette.py --file ../build_serial/su3_nd4_beta6_L8_L8_L8_L16_cfg_50.bin
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_CONFIG_NAME_RE = re.compile(
    r"^su(?P<ncolors>\d+)_nd(?P<ndims>\d+)_beta(?P<beta>[\d.eE+-]+)"
    r"(?P<latt>(?:_L\d+)+)_cfg_(?P<cfg>\d+)\.bin$"
)


@dataclass(frozen=True)
class ConfigMetadata:
    ncolors: int
    ndims: int
    beta: float
    lattice_size: tuple[int, ...]
    cfg_num: int


def parse_config_filename(path: str) -> ConfigMetadata:
    basename = os.path.basename(path)
    match = _CONFIG_NAME_RE.match(basename)
    if not match:
        raise ValueError(
            f"Cannot parse lattice parameters from filename '{basename}'. "
            "Expected KWQFT format, e.g. su3_nd4_beta6_L8_L8_L8_L16_cfg_50.bin"
        )

    lattice_size = tuple(
        int(length) for length in re.findall(r"_L(\d+)", match.group("latt"))
    )
    ndims = int(match.group("ndims"))
    if len(lattice_size) != ndims:
        raise ValueError(
            f"Filename lists {len(lattice_size)} lattice sizes but nd{ndims} "
            f"implies NDIMS={ndims}"
        )

    return ConfigMetadata(
        ncolors=int(match.group("ncolors")),
        ndims=ndims,
        beta=float(match.group("beta")),
        lattice_size=lattice_size,
        cfg_num=int(match.group("cfg")),
    )


def load_configuration(
    conf_path: str,
    lattice_size: Sequence[int],
    ncolors: int,
) -> np.ndarray:
    ndims = len(lattice_size)
    volume = int(np.prod(lattice_size))
    expected_bytes = volume * ndims * ncolors * ncolors * 16
    actual_bytes = os.path.getsize(conf_path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch for '{conf_path}': "
            f"got {actual_bytes} bytes, expected {expected_bytes} "
            f"(volume={volume}, ndims={ndims}, ncolors={ncolors})"
        )
    return np.fromfile(conf_path, dtype=np.complex128).reshape(
        (volume, ndims, ncolors, ncolors)
    )


def index_nd_nm(site_id: int, lattice_size: Sequence[int]) -> list[int]:
    x: list[int] = []
    temp = site_id
    for length in lattice_size:
        x.append(temp % length)
        temp //= length
    return x


def index_nd_nm_coord(x: Sequence[int], lattice_size: Sequence[int]) -> int:
    index = 0
    factor = 1
    for coord, length in zip(x, lattice_size):
        index += coord * factor
        factor *= length
    return index


def check_unitary(
    conf: np.ndarray,
    ncolors: int,
    prec: float = 1e-12,
) -> tuple[bool, float]:
    max_err = 0.0
    volume, ndims = conf.shape[0], conf.shape[1]
    identity = np.eye(ncolors, dtype=conf.dtype)
    for site in range(volume):
        for mu in range(ndims):
            unitary = conf[site, mu] @ conf[site, mu].conj().T
            err = float(np.max(np.abs(unitary - identity)))
            max_err = max(max_err, err)
    return max_err <= prec, max_err


def calc_plaq(
    conf: np.ndarray,
    lattice_size: Sequence[int],
    ncolors: int,
) -> tuple[float, float, float]:
    ndims = len(lattice_size)
    volume = int(np.prod(lattice_size))
    num_plaqs = ndims * (ndims - 1) // 2
    num_tplaqs = ndims - 1
    num_splaqs = num_plaqs - num_tplaqs

    plaq_s = 0.0
    plaq_t = 0.0
    for site in range(volume):
        x_org = index_nd_nm(site, lattice_size)
        for mu in range(ndims):
            xmu = x_org.copy()
            xmu[mu] = (xmu[mu] + 1) % lattice_size[mu]
            imu = index_nd_nm_coord(xmu, lattice_size)
            for nu in range(mu + 1, ndims):
                xnu = x_org.copy()
                xnu[nu] = (xnu[nu] + 1) % lattice_size[nu]
                inu = index_nd_nm_coord(xnu, lattice_size)
                plaq_multi = (
                    conf[site, mu]
                    @ conf[imu, nu]
                    @ conf[inu, mu].conj().T
                    @ conf[site, nu].conj().T
                )
                if nu == ndims - 1:
                    plaq_t += np.real(np.trace(plaq_multi))
                else:
                    plaq_s += np.real(np.trace(plaq_multi))

    plaq_s /= num_splaqs * ncolors * volume
    plaq_t /= num_tplaqs * ncolors * volume
    avg_plaq = (num_splaqs * plaq_s + num_tplaqs * plaq_t) / num_plaqs
    return float(plaq_s), float(plaq_t), float(avg_plaq)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate KWQFT binary gauge configurations."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to .bin configuration (KWQFT heatbath naming)",
    )
    parser.add_argument(
        "--unitary-prec",
        type=float,
        default=1e-12,
        help="Maximum allowed unitarity deviation (default: 1e-12)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conf_path = args.file

    if not os.path.isfile(conf_path):
        print(f"Error: gauge configuration file not found: {conf_path}", file=sys.stderr)
        return 1

    try:
        meta = parse_config_filename(conf_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    lattice_size = meta.lattice_size
    ndims = meta.ndims

    print(f"Configuration file : {conf_path}")
    print(f"Lattice            : {list(lattice_size)} (NDIMS={ndims})")
    print(
        f"SU(N)              : N={meta.ncolors}, beta={meta.beta}, "
        f"cfg={meta.cfg_num}"
    )
    print(
        "Expected file size : "
        f"{int(np.prod(lattice_size)) * ndims * meta.ncolors ** 2 * 16} bytes"
    )
    print()

    try:
        conf = load_configuration(conf_path, lattice_size, meta.ncolors)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    ok, max_err = check_unitary(conf, meta.ncolors, args.unitary_prec)
    if ok:
        print(f"Unitarity check    : PASSED (max error = {max_err:.3e})")
    else:
        print(
            f"Unitarity check    : FAILED (max error = {max_err:.3e}, "
            f"tolerance = {args.unitary_prec:.3e})",
            file=sys.stderr,
        )
        return 1

    plaq_s, plaq_t, avg_plaq = calc_plaq(conf, lattice_size, meta.ncolors)
    print(f"Spatial plaquette  : {plaq_s:.8f}")
    print(f"Temporal plaquette : {plaq_t:.8f}")
    print(f"Average plaquette  : {avg_plaq:.8f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
