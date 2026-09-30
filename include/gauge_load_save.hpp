/**
 * @file gauge_load_save.hpp
 * @brief Portable SOA / SOA12 gauge link load & store
 *
 * Layout stride is always the number of links (\c volume * NDIMS).
 * SOA12 (SU(3) only) stores the first two rows (6 complexes) and reconstructs
 * the third from unitarity; works on any Kokkos backend.
 */

#ifndef KWQFT_GAUGE_LOAD_SAVE_HPP
#define KWQFT_GAUGE_LOAD_SAVE_HPP

#include "complex.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"

namespace kwqft {

/// Number of complex elements stored per link for \p atype.
KOKKOS_INLINE_FUNCTION constexpr int gaugeComplexElems(ArrayType atype) {
  switch (atype) {
  case ArrayType::SOA12:
    return 6;
  case ArrayType::SOA8:
    return 4;
  case ArrayType::SOA:
  default:
    return NCOLORS * NCOLORS;
  }
}

/// Real parameters per link (for bandwidth accounting).
KOKKOS_INLINE_FUNCTION constexpr int gaugeNumParams(ArrayType atype) {
  return 2 * gaugeComplexElems(atype);
}

/**
 * @brief Reconstruct row 2 of an SU(3) matrix from the first two rows.
 *
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void reconstruct12p(MatrixSun<Real, 3> &a) {
  a.e[2][0] = ~(a.e[0][1] * a.e[1][2] - a.e[0][2] * a.e[1][1]);
  a.e[2][1] = ~(a.e[0][2] * a.e[1][0] - a.e[0][0] * a.e[1][2]);
  a.e[2][2] = ~(a.e[0][0] * a.e[1][1] - a.e[0][1] * a.e[1][0]);
}

/**
 * @brief Copy one Nc x Nc matrix whose element (i, j) lives at
 *        \p ptr[(j + i*Nc) * stride], optionally as its Hermitian conjugate.
 *
 * This is the single element-copy loop shared by every gauge/halo load path.
 * Keeping exactly one copy of the Nc^2 loop per call site (instead of one per
 * branch of the address resolution) is what keeps device code size, and
 * hence nvcc/cicc time, bounded for large Nc.
 *
 * \p ptr == nullptr yields the zero matrix (absent halo block).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void loadMatrixStrided(const Complex<Real> *ptr,
                                              int64_t stride, bool adjoint,
                                              MatrixSun<Real, NCOLORS> &u) {
  if (ptr == nullptr) {
    u = MatrixSun<Real, NCOLORS>::zero();
    return;
  }
  if (adjoint) {
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        u.e[j][i] = ~ptr[(j + i * NCOLORS) * stride];
      }
    }
  } else {
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        u.e[i][j] = ptr[(j + i * NCOLORS) * stride];
      }
    }
  }
}

/**
 * @brief Load one link matrix from gauge storage at link base index.
 *
 * \p link_base = idx_eo + dir * volume (EO layout).
 * \p soa_stride = number of links (typically volume * NDIMS).
 * SOA12 is only a distinct layout for Nc == 3; for any other Nc every
 * ArrayType is a full SOA matrix and there is a single code path.
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGaugeMatrix(const Complex<Real> *ptr, int64_t link_base, int64_t soa_stride,
                ArrayType atype, MatrixSun<Real, NCOLORS> &u,
                bool adjoint = false) {
  if constexpr (NCOLORS == 3) {
    if (atype == ArrayType::SOA12) {
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
          u.e[i][j] = ptr[link_base + (j + i * 3) * soa_stride];
        }
      }
      reconstruct12p(u);
      if (adjoint) {
        u = u.dagger();
      }
      return;
    }
  }
  (void)atype;
  loadMatrixStrided(ptr + link_base, soa_stride, adjoint, u);
}

/**
 * @brief Store one link matrix into gauge storage (SOA12 writes two rows only).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
storeGaugeMatrix(Complex<Real> *ptr, int64_t link_base, int64_t soa_stride,
                 ArrayType atype, const MatrixSun<Real, NCOLORS> &u) {
  if constexpr (NCOLORS == 3) {
    if (atype == ArrayType::SOA12) {
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
          ptr[link_base + (j + i * 3) * soa_stride] = u.e[i][j];
        }
      }
      return;
    }
  }
  (void)atype;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      ptr[link_base + (j + i * NCOLORS) * soa_stride] = u.e[i][j];
    }
  }
}

} // namespace kwqft

#endif // KWQFT_GAUGE_LOAD_SAVE_HPP
