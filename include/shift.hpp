/**
 * @file shift.hpp
 * @brief EO index shift and SOA gauge link load helpers
 */

#ifndef KWQFT_SHIFT_HPP
#define KWQFT_SHIFT_HPP

#include "constants.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"

namespace kwqft {

/// Chroma/QDP-compatible shift directions for \ref shift_eo / lazy \c shift().
enum ShiftDirection : int {
  BACKWARD = -1,
  FORWARD = 1,
};

/**
 * @brief Shift an even/odd site index by ±e_mu (periodic on the local grid).
 */
KOKKOS_INLINE_FUNCTION int64_t shift_eo(int64_t idx_eo, int mu, int lmu,
                                        const LatticeParams &p) {
  const int oddbit = (idx_eo >= p.half_volume) ? 1 : 0;
  const int64_t id = idx_eo - static_cast<int64_t>(oddbit) * p.half_volume;
  return indexNdNeigEo(id, oddbit, mu, lmu, p);
}

/**
 * @brief Load one SU(N) link from SOA gauge storage (even/odd layout).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGaugeLinkSoa(const Complex<Real> *gaugePtr, int64_t idx_eo, int dir,
                 int64_t soa_stride, const LatticeParams &p,
                 MatrixSun<Real, NCOLORS> &U) {
  const int64_t base = idx_eo + static_cast<int64_t>(dir) * p.volume;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      U.e[i][j] = gaugePtr[base + (j + i * NCOLORS) * soa_stride];
    }
  }
}

} // namespace kwqft

#endif // KWQFT_SHIFT_HPP
