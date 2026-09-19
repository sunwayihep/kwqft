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
KOKKOS_INLINE_FUNCTION constexpr int
gauge_complex_elems(ArrayType atype) {
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
KOKKOS_INLINE_FUNCTION constexpr int gauge_num_params(ArrayType atype) {
  return 2 * gauge_complex_elems(atype);
}

/**
 * @brief Reconstruct row 2 of an SU(3) matrix from the first two rows.
 *
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void reconstruct12p(MatrixSun<Real, 3> &A) {
  A.e[2][0] = ~(A.e[0][1] * A.e[1][2] - A.e[0][2] * A.e[1][1]);
  A.e[2][1] = ~(A.e[0][2] * A.e[1][0] - A.e[0][0] * A.e[1][2]);
  A.e[2][2] = ~(A.e[0][0] * A.e[1][1] - A.e[0][1] * A.e[1][0]);
}

/**
 * @brief Load one link matrix from gauge storage at link base index.
 *
 * \p link_base = idx_eo + dir * volume (EO layout).
 * \p soa_stride = number of links (typically volume * NDIMS).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGaugeMatrix(const Complex<Real> *ptr, int64_t link_base, int64_t soa_stride,
                ArrayType atype, MatrixSun<Real, NCOLORS> &U) {
  if (atype == ArrayType::SOA12) {
#if NCOLORS == 3
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        U.e[i][j] = ptr[link_base + (j + i * 3) * soa_stride];
      }
    }
    reconstruct12p(U);
#else
    // Unreachable if callers guard SOA12 to Nc==3; keep a safe fallback.
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        U.e[i][j] = ptr[link_base + (j + i * NCOLORS) * soa_stride];
      }
    }
#endif
  } else {
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        U.e[i][j] = ptr[link_base + (j + i * NCOLORS) * soa_stride];
      }
    }
  }
}

/**
 * @brief Store one link matrix into gauge storage (SOA12 writes two rows only).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
storeGaugeMatrix(Complex<Real> *ptr, int64_t link_base, int64_t soa_stride,
                 ArrayType atype, const MatrixSun<Real, NCOLORS> &U) {
  if (atype == ArrayType::SOA12) {
#if NCOLORS == 3
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        ptr[link_base + (j + i * 3) * soa_stride] = U.e[i][j];
      }
    }
#else
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        ptr[link_base + (j + i * NCOLORS) * soa_stride] = U.e[i][j];
      }
    }
#endif
  } else {
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        ptr[link_base + (j + i * NCOLORS) * soa_stride] = U.e[i][j];
      }
    }
  }
}

} // namespace kwqft

#endif // KWQFT_GAUGE_LOAD_SAVE_HPP
