/**
 * @file neighbor_access.hpp
 * @brief Coordinate-based gauge link load (periodic serial or MPI + halo)
 */

#ifndef KWQFT_NEIGHBOR_ACCESS_HPP
#define KWQFT_NEIGHBOR_ACCESS_HPP

#include "constants.hpp"
#include "index.hpp"
#include "matrixsun.hpp"
#include "shift.hpp"

namespace kwqft {

constexpr int halo_pow3_constexpr(int n) {
  return (n <= 0) ? 1 : 3 * halo_pow3_constexpr(n - 1);
}
constexpr int HALO_CODE_COUNT = halo_pow3_constexpr(NDIMS);
constexpr int HALO_CENTER_CODE = (HALO_CODE_COUNT - 1) / 2;

KOKKOS_INLINE_FUNCTION int halo_offset_to_code(const int off[NDIMS]) {
  int code = 0;
  int mult = 1;
  for (int d = 0; d < NDIMS; ++d) {
    code += (off[d] + 1) * mult;
    mult *= 3;
  }
  return code;
}

KOKKOS_INLINE_FUNCTION void halo_code_to_offset(int code, int off[NDIMS]) {
  int t = code;
  for (int d = 0; d < NDIMS; ++d) {
    const int digit = t % 3;
    off[d] = digit - 1;
    t /= 3;
  }
}

KOKKOS_INLINE_FUNCTION int64_t halo_region_volume(const int off[NDIMS],
                                                  const LatticeParams &p) {
  int64_t vol = 1;
  for (int d = 0; d < NDIMS; ++d) {
    vol *= (off[d] == 0) ? static_cast<int64_t>(p.grid[d]) : 1LL;
  }
  return vol;
}

/**
 * @brief Halo region parity split.
 *
 * A region can be stored parity-major (all even sites, then all odd sites)
 * when the extent of its first free dimension is even; then one
 * (dir, parity) block is contiguous and can be exchanged alone after a
 * checkerboard update. Returns the first free dimension, or -1 if the region
 * cannot be split (odd first free extent; region is then stored site-major).
 */
KOKKOS_INLINE_FUNCTION int halo_region_split_dim(const int off[NDIMS],
                                                 const LatticeParams &p) {
  for (int d = 0; d < NDIMS; ++d) {
    if (off[d] == 0) {
      return (p.grid[d] % 2 == 0) ? d : -1;
    }
  }
  return -1;
}

/**
 * @brief Slot of site \p x inside halo region \p off (0 <= slot < region vol).
 *
 * Parity-split regions: slot = parity * vol/2 + x[d0]/2 + (g0/2) * rest,
 * where \p rest is the mixed-radix index over the remaining free dims.
 * Non-split: plain mixed-radix index over all free dims.
 */
KOKKOS_INLINE_FUNCTION int64_t halo_region_slot(const int off[NDIMS],
                                                const int x[NDIMS],
                                                int split_dim, int64_t vol,
                                                const LatticeParams &p) {
  if (split_dim < 0) {
    int64_t idx = 0;
    int64_t mult = 1;
    for (int d = 0; d < NDIMS; ++d) {
      if (off[d] != 0) {
        continue;
      }
      idx += static_cast<int64_t>(x[d]) * mult;
      mult *= static_cast<int64_t>(p.grid[d]);
    }
    return idx;
  }
  int parity = 0;
  int64_t rest = 0;
  int64_t mult = 1;
  for (int d = 0; d < NDIMS; ++d) {
    parity += x[d];
    if (off[d] != 0 || d == split_dim) {
      continue;
    }
    rest += static_cast<int64_t>(x[d]) * mult;
    mult *= static_cast<int64_t>(p.grid[d]);
  }
  parity &= 1;
  const int64_t half0 = static_cast<int64_t>(p.grid[split_dim] / 2);
  return static_cast<int64_t>(parity) * (vol / 2) +
         static_cast<int64_t>(x[split_dim] / 2) + half0 * rest;
}

/**
 * @brief Inverse of \ref halo_region_slot: site coords from a region slot.
 *
 * Coordinates on the fixed (off != 0) dims are set to the *local* face value
 * (0 for +1, grid-1 for -1), i.e. the sender-side convention used by pack.
 */
KOKKOS_INLINE_FUNCTION void halo_slot_to_coords(const int off[NDIMS],
                                                int split_dim, int64_t vol,
                                                int64_t slot, int x[NDIMS],
                                                const LatticeParams &p) {
  if (split_dim < 0) {
    int64_t t = slot;
    for (int d = 0; d < NDIMS; ++d) {
      if (off[d] == -1) {
        x[d] = p.grid[d] - 1;
      } else if (off[d] == +1) {
        x[d] = 0;
      } else {
        x[d] = static_cast<int>(t % static_cast<int64_t>(p.grid[d]));
        t /= static_cast<int64_t>(p.grid[d]);
      }
    }
    return;
  }
  const int64_t half = vol / 2;
  const int parity = (slot >= half) ? 1 : 0;
  int64_t t = slot - static_cast<int64_t>(parity) * half;
  const int64_t half0 = static_cast<int64_t>(p.grid[split_dim] / 2);
  const int h = static_cast<int>(t % half0);
  t /= half0;
  int others = 0;
  for (int d = 0; d < NDIMS; ++d) {
    if (off[d] == -1) {
      x[d] = p.grid[d] - 1;
    } else if (off[d] == +1) {
      x[d] = 0;
    } else if (d == split_dim) {
      x[d] = 0; // filled below
      continue;
    } else {
      x[d] = static_cast<int>(t % static_cast<int64_t>(p.grid[d]));
      t /= static_cast<int64_t>(p.grid[d]);
    }
    others += x[d];
  }
  x[split_dim] = 2 * h + ((parity + others) & 1);
}

/// Halo recv buffers: per code, element index is (dir * vol + slot) * (Nc*Nc)
/// with \p slot from \ref halo_region_slot, so one link direction (and, for
/// parity-split regions, one (dir, parity) block) is contiguous.
template <typename Real> struct GaugeHaloDevice {
  const Complex<Real> *recv[HALO_CODE_COUNT]{};
  int64_t vol[HALO_CODE_COUNT]{};
  int split_dim[HALO_CODE_COUNT]{};
};

template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGhostFaceLink(const Complex<Real> *buf, int64_t face_vol, int64_t slot,
                  int dir, MatrixSun<Real, NCOLORS> &U) {
  if (buf == nullptr) {
    U = MatrixSun<Real, NCOLORS>::zero();
    return;
  }
  const int64_t me = static_cast<int64_t>(NCOLORS * NCOLORS);
  const int64_t off = (static_cast<int64_t>(dir) * face_vol + slot) * me;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      U.e[i][j] = buf[off + j + i * NCOLORS];
    }
  }
}

KOKKOS_INLINE_FUNCTION void eo_to_coords(int64_t id, int oddbit, int x[NDIMS],
                                         const LatticeParams &p) {
  indexNdEo(x, id, oddbit, p);
}

/**
 * @brief Load link U_dir at integer site coords.
 *
 * Serial (!p.mpi): periodic wrap.
 * MPI: dimensions with \c proc_grid[d]==1 wrap locally; only true subdomain
 * boundaries use \p halo (must be exchanged first).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGaugeLinkAtCoords(const Complex<Real> *gaugePtr, int64_t soa_stride,
                      const GaugeHaloDevice<Real> *halo, const int x[NDIMS],
                      int dir, const LatticeParams &p,
                      MatrixSun<Real, NCOLORS> &U,
                      ArrayType atype = ArrayType::SOA) {
  int off[NDIMS];
  int xw[NDIMS];
  bool need_halo = false;

  for (int d = 0; d < NDIMS; ++d) {
    const int g = p.grid[d];
    // Not domain-decomposed: always local periodic (no MPI neighbor).
    if (!p.mpi || p.proc_grid[d] <= 1) {
      int v = x[d] % g;
      if (v < 0) {
        v += g;
      }
      xw[d] = v;
      off[d] = 0;
      continue;
    }
    if (x[d] < 0) {
      off[d] = -1;
      xw[d] = g - 1;
      need_halo = true;
    } else if (x[d] >= g) {
      off[d] = +1;
      xw[d] = 0;
      need_halo = true;
    } else {
      off[d] = 0;
      xw[d] = x[d];
    }
  }

  if (!need_halo) {
    const int64_t idx_eo = coords_to_eo_idx(xw, p);
    loadGaugeLinkSoa(gaugePtr, idx_eo, dir, soa_stride, p, U, atype);
    return;
  }

  if (halo == nullptr) {
    U = MatrixSun<Real, NCOLORS>::zero();
    return;
  }

  const int code = halo_offset_to_code(off);
  const Complex<Real> *buf = halo->recv[code];
  if (buf == nullptr) {
    U = MatrixSun<Real, NCOLORS>::zero();
    return;
  }

  const int64_t vol = halo->vol[code];
  const int64_t slot =
      halo_region_slot(off, xw, halo->split_dim[code], vol, p);
  loadGhostFaceLink(buf, vol, slot, dir, U);
}

} // namespace kwqft

#endif
