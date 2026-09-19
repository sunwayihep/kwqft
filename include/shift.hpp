/**
 * @file shift.hpp
 * @brief EO index shift and SOA / SOA12 gauge link load helpers
 */

#ifndef KWQFT_SHIFT_HPP
#define KWQFT_SHIFT_HPP

#include "constants.hpp"
#include "gauge_load_save.hpp"
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
 * @brief Load one SU(N) link from SOA / SOA12 gauge storage (even/odd layout).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGaugeLinkSoa(const Complex<Real> *gaugePtr, int64_t idx_eo, int dir,
                 int64_t soa_stride, const LatticeParams &p,
                 MatrixSun<Real, NCOLORS> &U,
                 ArrayType atype = ArrayType::SOA) {
  const int64_t base = idx_eo + static_cast<int64_t>(dir) * p.volume;
  loadGaugeMatrix(gaugePtr, base, soa_stride, atype, U);
}

/**
 * @brief Store one SU(N) link into SOA / SOA12 gauge storage.
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
storeGaugeLinkSoa(Complex<Real> *gaugePtr, int64_t idx_eo, int dir,
                  int64_t soa_stride, const LatticeParams &p,
                  const MatrixSun<Real, NCOLORS> &U,
                  ArrayType atype = ArrayType::SOA) {
  const int64_t base = idx_eo + static_cast<int64_t>(dir) * p.volume;
  storeGaugeMatrix(gaugePtr, base, soa_stride, atype, U);
}

} // namespace kwqft

#endif // KWQFT_SHIFT_HPP
