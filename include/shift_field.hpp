/**
 * @file shift_field.hpp
 * @brief QDPXX-style lattice fields with lazy shift/adjoint views
 *
 * \c shift() / \c adj() only update metadata (no full-volume copies).
 * Values are resolved by \ref LatticeColorMatrix::load_at:
 *   - local / non-split dims: EO index arithmetic into SOA (zero copy)
 *   - MPI domain boundary: read from face/edge halo buffers only
 */

#ifndef KWQFT_SHIFT_FIELD_HPP
#define KWQFT_SHIFT_FIELD_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"
#include "neighbor_access.hpp"
#include "shift.hpp"

namespace kwqft {

constexpr int gauge_matrix_elems() { return NCOLORS * NCOLORS; }

/// Max composed nearest-neighbor shifts on a lazy view (staple needs 2).
constexpr int LCM_MAX_SHIFTS = 4;

/**
 * @brief One SU(N) matrix per site (QDP \c LatticeColorMatrix).
 *
 * Lightweight view into SOA gauge storage, optionally with a shift chain and
 * adjoint flag.
 */
template <typename Real> class LatticeColorMatrix {
public:
  using value_type = Real;
  using ComplexT = Complex<Real>;
  using MatrixT = MatrixSun<Real, NCOLORS>;
  static constexpr int site_elems = gauge_matrix_elems();

  LatticeColorMatrix() = default;

  KOKKOS_INLINE_FUNCTION
  static LatticeColorMatrix gauge_soa(const ComplexT *base, int64_t stride,
                                      int mu) {
    LatticeColorMatrix f;
    f.data_ = base;
    f.stride_ = stride;
    f.link_dir_ = mu;
    return f;
  }

  KOKKOS_INLINE_FUNCTION const ComplexT *data() const { return data_; }
  KOKKOS_INLINE_FUNCTION int64_t stride() const { return stride_; }
  KOKKOS_INLINE_FUNCTION int link_dir() const { return link_dir_; }

  KOKKOS_INLINE_FUNCTION
  LatticeColorMatrix with_shift(ShiftDirection dir, int mu) const {
    LatticeColorMatrix out = *this;
    if (out.n_shifts_ >= LCM_MAX_SHIFTS) {
      return out;
    }
    out.shift_mu_[out.n_shifts_] = mu;
    out.shift_sign_[out.n_shifts_] = static_cast<int>(dir);
    out.n_shifts_++;
    return out;
  }

  KOKKOS_INLINE_FUNCTION
  LatticeColorMatrix with_adjoint() const {
    LatticeColorMatrix out = *this;
    out.adjoint_ = !out.adjoint_;
    return out;
  }

  /**
   * @brief Evaluate at EO site \p idx_eo.
   *
   * No field copies. Interior / serial: \ref shift_eo + SOA load.
   * MPI: only sites whose shift chain crosses a split-domain face use \p halo.
   */
  KOKKOS_INLINE_FUNCTION
  void load_at(int64_t idx_eo, const LatticeParams &p,
               const GaugeHaloDevice<Real> *halo, MatrixT &U) const {
    // Fast path: every shift direction is local (no MPI split) → EO wrap only.
    bool local_eo = true;
    if (p.mpi) {
      for (int s = 0; s < n_shifts_; ++s) {
        if (p.proc_grid[shift_mu_[s]] > 1) {
          local_eo = false;
          break;
        }
      }
    }

    if (local_eo) {
      int64_t idx = idx_eo;
      for (int s = 0; s < n_shifts_; ++s) {
        idx = shift_eo(idx, shift_mu_[s], shift_sign_[s], p);
      }
      loadGaugeLinkSoa(data_, idx, link_dir_, stride_, p, U);
    } else {
      // May hit a subdomain face: coordinate path uses halo only if out of range.
      const int oddbit = (idx_eo >= p.half_volume) ? 1 : 0;
      const int64_t id = idx_eo - static_cast<int64_t>(oddbit) * p.half_volume;
      int x[NDIMS];
      eo_to_coords(id, oddbit, x, p);
      for (int s = 0; s < n_shifts_; ++s) {
        x[shift_mu_[s]] += shift_sign_[s];
      }
      loadGaugeLinkAtCoords(data_, stride_, halo, x, link_dir_, p, U);
    }

    if (adjoint_) {
      U = U.dagger();
    }
  }

private:
  const ComplexT *data_{nullptr};
  int64_t stride_{0};
  int link_dir_{0};
  bool adjoint_{false};
  int n_shifts_{0};
  int shift_mu_[LCM_MAX_SHIFTS]{};
  int shift_sign_[LCM_MAX_SHIFTS]{};
};

/// All Nd link directions (QDP \c multi1d<LatticeColorMatrix> \c u).
template <typename Real> class LatticeGaugeLinks {
public:
  using ComplexT = Complex<Real>;
  using MatrixT = LatticeColorMatrix<Real>;

  LatticeGaugeLinks() = default;
  KOKKOS_INLINE_FUNCTION
  LatticeGaugeLinks(const ComplexT *data, int64_t soa_stride)
      : data_(data), stride_(soa_stride) {}

  KOKKOS_INLINE_FUNCTION const ComplexT *data() const { return data_; }
  KOKKOS_INLINE_FUNCTION int64_t stride() const { return stride_; }

  KOKKOS_INLINE_FUNCTION
  MatrixT operator[](int mu) const {
    return MatrixT::gauge_soa(data_, stride_, mu);
  }

private:
  const ComplexT *data_{nullptr};
  int64_t stride_{0};
};

/// Lazy QDP-style shift: \c result(x) = field(x + dir * e_mu).  No data move.
template <typename Real>
KOKKOS_INLINE_FUNCTION LatticeColorMatrix<Real>
shift(const LatticeColorMatrix<Real> &field, ShiftDirection dir, int mu) {
  return field.with_shift(dir, mu);
}

} // namespace kwqft

#endif
