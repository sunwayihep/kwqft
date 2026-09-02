/**
 * @file shift.hpp
 * @brief Lattice shift helpers (single-process copy semantics)
 *
 * Mirrors QDPXX \c shift(field, dir, mu) with
 *   dest(x) = src(x + isign * e_mu)   (isign = FORWARD/BACKWARD)
 * FORWARD means the source site is in the +mu direction from the destination.
 *
 * Storage uses even/odd (checkerboard) indices \c idx_eo in [0, volume).
 * MPI subdomain shifts use the global \ref shiftMap with a \ref ShiftLayout
 * descriptor (initialized by \ref initializeParams).
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
  BACKWARD = -1,
  FORWARD = 1,
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
 * @brief Single-process bulk forward shift of a dense EO field.
 *
 * Fills \c dst so that \c dst(x) = \c src(x + e_mu), i.e.
 * QDPXX \c shift(src, FORWARD, mu).
 */
template <typename SrcView, typename DstView>
void shift_field_forward_eo(const SrcView &src, const DstView &dst, int mu,
                            int site_elems, const LatticeParams &p) {
  const int64_t vol = p.volume;
  const int64_t se = static_cast<int64_t>(site_elems);
  Kokkos::parallel_for(
      "shift_field_forward_eo",
      Kokkos::RangePolicy<DefaultExecSpace>(0, vol), KOKKOS_LAMBDA(int64_t idx_eo) {
        const int64_t src_idx = shift_eo(idx_eo, mu, FORWARD, p);
        for (int64_t k = 0; k < se; ++k) {
          dst(idx_eo * se + k) = src(src_idx * se + k);
        }
      });
  Kokkos::fence();
}

/**
 * @brief Single-process bulk backward shift of a dense EO field.
 */
template <typename SrcView, typename DstView>
void shift_field_backward_eo(const SrcView &src, const DstView &dst, int mu,
                             int site_elems, const LatticeParams &p) {
  const int64_t vol = p.volume;
  const int64_t se = static_cast<int64_t>(site_elems);
  Kokkos::parallel_for(
      "shift_field_backward_eo",
      Kokkos::RangePolicy<DefaultExecSpace>(0, vol), KOKKOS_LAMBDA(int64_t idx_eo) {
        const int64_t src_idx = shift_eo(idx_eo, mu, BACKWARD, p);
        for (int64_t k = 0; k < se; ++k) {
          dst(idx_eo * se + k) = src(src_idx * se + k);
        }
      });
  Kokkos::fence();
}

/// Gauge link forward shift (dense EO, one matrix per site).
template <typename SrcView, typename DstView>
void shift_link_field_forward_eo(const SrcView &src, const DstView &dst, int mu,
                                 const LatticeParams &p) {
  shift_field_forward_eo(src, dst, mu, NCOLORS * NCOLORS, p);
}

/// Gauge link backward shift (dense EO, one matrix per site).
template <typename SrcView, typename DstView>
void shift_link_field_backward_eo(const SrcView &src, const DstView &dst, int mu,
                                  const LatticeParams &p) {
  shift_field_backward_eo(src, dst, mu, NCOLORS * NCOLORS, p);
}

} // namespace kwqft

#endif // KWQFT_SHIFT_HPP
