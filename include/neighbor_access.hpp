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

/// Halo recv buffers: layout per code is dir-major,
/// index (dir * vol + face_idx) * (Nc*Nc), so one link direction is contiguous.
template <typename Real> struct GaugeHaloDevice {
  const Complex<Real> *recv[HALO_CODE_COUNT]{};
  int64_t vol[HALO_CODE_COUNT]{};
};

template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGhostFaceLink(const Complex<Real> *buf, int64_t face_vol, int64_t face_idx,
                  int dir, MatrixSun<Real, NCOLORS> &U) {
  if (buf == nullptr) {
    U = MatrixSun<Real, NCOLORS>::zero();
    return;
  }
  const int64_t me = static_cast<int64_t>(NCOLORS * NCOLORS);
  const int64_t off =
      (static_cast<int64_t>(dir) * face_vol + face_idx) * me;
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

  int64_t region_idx = 0;
  int64_t mult = 1;
  for (int d = 0; d < NDIMS; ++d) {
    const int c = (off[d] == 0) ? xw[d] : 0;
    region_idx += static_cast<int64_t>(c) * mult;
    mult *= (off[d] == 0) ? static_cast<int64_t>(p.grid[d]) : 1LL;
  }
  loadGhostFaceLink(buf, halo->vol[code], region_idx, dir, U);
}

} // namespace kwqft

#endif
