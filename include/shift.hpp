/**
 * @file shift.hpp
 * @brief Lattice shift helpers (single-process copy semantics)
 *
 * Mirrors QDP/Chroma-style \c shift(field, dir, mu): on a site x,
 *   (shift(F, FORWARD,  mu))(x) = F(x + e_mu)
 *   (shift(F, BACKWARD, mu))(x) = F(x - e_mu)
 *
 * Storage uses even/odd (checkerboard) indices \c idx_eo in [0, volume),
 * matching \c indexNdNeigEo. A future MPI implementation can replace the
 * index mapping with halo exchange + local gather (cf. QDP \c Map).
 */

#ifndef KWQFT_SHIFT_HPP
#define KWQFT_SHIFT_HPP

#include "constants.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"

namespace kwqft {

/// Chroma/QDP-compatible shift directions for \ref shift_eo.
enum ShiftDirection : int {
  SHIFT_BACKWARD = -1,
  SHIFT_FORWARD = 1,
};

/**
 * @brief Shift an even/odd site index by ±e_mu (periodic).
 *
 * @param idx_eo  EO storage index (0 … volume-1): id + oddbit * half_volume
 * @param mu      direction 0 … NDIMS-1
 * @param lmu     +1 (FORWARD) or -1 (BACKWARD)
 */
KOKKOS_INLINE_FUNCTION int64_t shift_eo(int64_t idx_eo, int mu, int lmu,
                                        const LatticeParams &p) {
  const int oddbit = (idx_eo >= p.half_volume) ? 1 : 0;
  const int64_t id = idx_eo - static_cast<int64_t>(oddbit) * p.half_volume;
  return indexNdNeigEo(id, oddbit, mu, lmu, p);
}

/**
 * @brief Load one SU(N) link from SOA gauge storage (even/odd layout).
 *
 * @param gaugePtr   Raw gauge array (same layout as \c GaugeArray SOA)
 * @param idx_eo     Site index in EO encoding
 * @param dir        Link direction mu
 * @param soa_stride Stride between matrix elements = volume * NDIMS (per-link
 *                   slot count in the SOA layout)
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
loadGaugeLinkSoa(const Complex<Real> *gaugePtr, int64_t idx_eo, int dir,
                 int64_t soa_stride, const LatticeParams &p,
                 MatrixSun<Real, NCOLORS> &U) {
  const int64_t base =
      idx_eo + static_cast<int64_t>(dir) * p.volume;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      U.e[i][j] = gaugePtr[base + (j + i * NCOLORS) * soa_stride];
    }
  }
}

/**
 * @brief Single-process bulk forward shift of one direction's links.
 *
 * Fills \c dst so that \c dst(x) = \c src(shift(x, FORWARD, mu))
 *         = \c src(x + e_mu) in EO labeling, matching QDP/Chroma \c shift(., FORWARD, mu).
 *
 * Both views must have length \c p.volume * NCOLORS * NCOLORS (one matrix per
 * EO site, dense layout). This is a building block for a future MPI halo path
 * where \c shift_eo is replaced by communicator gathers.
 */
template <typename SrcView, typename DstView>
void shift_link_field_forward_eo(const SrcView &src, const DstView &dst, int mu,
                                 const LatticeParams &p) {
  const int64_t vol = p.volume;
  const int64_t mat_elems = static_cast<int64_t>(NCOLORS * NCOLORS);
  Kokkos::parallel_for(
      "shift_link_field_forward_eo",
      Kokkos::RangePolicy<DefaultExecSpace>(0, vol), KOKKOS_LAMBDA(int64_t idx_eo) {
        const int64_t src_idx = shift_eo(idx_eo, mu, SHIFT_FORWARD, p);
        for (int64_t k = 0; k < mat_elems; ++k) {
          dst(idx_eo * mat_elems + k) = src(src_idx * mat_elems + k);
        }
      });
  Kokkos::fence();
}

/**
 * @brief Single-process bulk backward shift of one direction's links.
 *
 * \c dst(x) = \c src(x - e_mu) = \c src(shift_eo(x, mu, BACKWARD)).
 */
template <typename SrcView, typename DstView>
void shift_link_field_backward_eo(const SrcView &src, const DstView &dst, int mu,
                                  const LatticeParams &p) {
  const int64_t vol = p.volume;
  const int64_t mat_elems = static_cast<int64_t>(NCOLORS * NCOLORS);
  Kokkos::parallel_for(
      "shift_link_field_backward_eo",
      Kokkos::RangePolicy<DefaultExecSpace>(0, vol), KOKKOS_LAMBDA(int64_t idx_eo) {
        const int64_t src_idx = shift_eo(idx_eo, mu, SHIFT_BACKWARD, p);
        for (int64_t k = 0; k < mat_elems; ++k) {
          dst(idx_eo * mat_elems + k) = src(src_idx * mat_elems + k);
        }
      });
  Kokkos::fence();
}

} // namespace kwqft

#endif // KWQFT_SHIFT_HPP
