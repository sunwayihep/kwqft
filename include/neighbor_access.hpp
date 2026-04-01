/**
 * @file neighbor_access.hpp
 * @brief Site-coordinate gauge link access (periodic serial or subdomain + halo MPI)
 */

#ifndef KWQFT_NEIGHBOR_ACCESS_HPP
#define KWQFT_NEIGHBOR_ACCESS_HPP

#include "constants.hpp"
#include "matrixsun.hpp"
#include "shift.hpp"

namespace kwqft {

template <typename Real> struct GaugeHaloDevice {
  const Complex<Real> *recv_m[NDIMS]{};
  const Complex<Real> *recv_p[NDIMS]{};
};

KOKKOS_INLINE_FUNCTION void eo_to_coords(int64_t id, int oddbit, int x[NDIMS],
                                         const LatticeParams &p) {
  indexNdEo(x, id, oddbit, p);
}

KOKKOS_INLINE_FUNCTION int64_t coords_to_eo_idx(const int x[NDIMS],
                                                const LatticeParams &p) {
  int64_t pos = 0;
  int64_t factor = 1;
  for (int i = 0; i < NDIMS; ++i) {
    pos += static_cast<int64_t>(x[i]) * factor;
    factor *= static_cast<int64_t>(p.grid[i]);
  }
  pos /= 2;
  int sumX = 0;
  for (int i = 0; i < NDIMS; ++i) {
    sumX += x[i];
  }
  const int oddbit1 = sumX & 1;
  pos += static_cast<int64_t>(oddbit1) * p.half_volume;
  return pos;
}

/// Inverse of \ref face_idx_tangential for the face `x[mu] = L[mu]-1`.
KOKKOS_INLINE_FUNCTION void fidx_to_coords_high_face(int64_t fidx, int mu,
                                                     int x[NDIMS],
                                                     const LatticeParams &p) {
  int64_t t = fidx;
  for (int d = 0; d < NDIMS; ++d) {
    if (d == mu) {
      x[d] = p.grid[d] - 1;
    } else {
      x[d] = static_cast<int>(t % static_cast<int64_t>(p.grid[d]));
      t /= static_cast<int64_t>(p.grid[d]);
    }
  }
}

/// Face `x[mu] = 0` with same tangential ordering as \ref face_idx_tangential.
KOKKOS_INLINE_FUNCTION void fidx_to_coords_low_face(int64_t fidx, int mu,
                                                  int x[NDIMS],
                                                  const LatticeParams &p) {
  int64_t t = fidx;
  for (int d = 0; d < NDIMS; ++d) {
    if (d == mu) {
      x[d] = 0;
    } else {
      x[d] = static_cast<int>(t % static_cast<int64_t>(p.grid[d]));
      t /= static_cast<int64_t>(p.grid[d]);
    }
  }
}

KOKKOS_INLINE_FUNCTION int64_t
face_idx_tangential(const int x[NDIMS], int mu, const LatticeParams &p) {
  int64_t idx = 0;
  int64_t mult = 1;
  for (int d = 0; d < NDIMS; ++d) {
    if (d == mu) {
      continue;
    }
    idx += static_cast<int64_t>(x[d]) * mult;
    mult *= static_cast<int64_t>(p.grid[d]);
  }
  return idx;
}

template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGhostFaceLink(const GaugeHaloDevice<Real> &h, int mu_bnd, bool plus_face,
                  int64_t face_idx, int dir, MatrixSun<Real, NCOLORS> &U) {
  const Complex<Real> *buf = plus_face ? h.recv_p[mu_bnd] : h.recv_m[mu_bnd];
  const int64_t me = static_cast<int64_t>(NCOLORS * NCOLORS);
  const int64_t off = (face_idx * NDIMS + dir) * me;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      U.e[i][j] = buf[off + j + i * NCOLORS];
    }
  }
}

/**
 * @brief Load link U_dir at site x (integer coords; may be outside local box for MPI).
 *
 * Serial (\c p.mpi false): periodic wrap.
 * MPI: \c halo must be non-null; uses ghost buffers for out-of-range coords.
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGaugeLinkAtCoords(const Complex<Real> *gaugePtr, int64_t soa_stride,
                      const GaugeHaloDevice<Real> *halo, const int x[NDIMS],
                      int dir, const LatticeParams &p,
                      MatrixSun<Real, NCOLORS> &U) {
  if (!p.mpi) {
    int xw[NDIMS];
    for (int d = 0; d < NDIMS; ++d) {
      int g = p.grid[d];
      int v = x[d] % g;
      if (v < 0) {
        v += g;
      }
      xw[d] = v;
    }
    const int64_t idx_eo = coords_to_eo_idx(xw, p);
    loadGaugeLinkSoa(gaugePtr, idx_eo, dir, soa_stride, p, U);
    return;
  }

  bool all_in = true;
  for (int d = 0; d < NDIMS; ++d) {
    if (x[d] < 0 || x[d] >= p.grid[d]) {
      all_in = false;
      break;
    }
  }
  if (all_in) {
    const int64_t idx_eo = coords_to_eo_idx(x, p);
    loadGaugeLinkSoa(gaugePtr, idx_eo, dir, soa_stride, p, U);
    return;
  }

  if (halo == nullptr) {
    U = MatrixSun<Real, NCOLORS>::zero();
    return;
  }

  for (int mu = 0; mu < NDIMS; ++mu) {
    if (x[mu] < 0) {
      int xc[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        xc[d] = x[d];
      }
      xc[mu] = 0;
      const int64_t face_idx = face_idx_tangential(xc, mu, p);
      loadGhostFaceLink(*halo, mu, false, face_idx, dir, U);
      return;
    }
    if (x[mu] >= p.grid[mu]) {
      int xc[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        xc[d] = x[d];
      }
      xc[mu] = p.grid[mu] - 1;
      const int64_t face_idx = face_idx_tangential(xc, mu, p);
      loadGhostFaceLink(*halo, mu, true, face_idx, dir, U);
      return;
    }
  }
  U = MatrixSun<Real, NCOLORS>::zero();
}

} // namespace kwqft

#endif
