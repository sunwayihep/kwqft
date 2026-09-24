/**
 * @file shift_field.hpp
 * @brief QDPXX-style lattice fields with lazy shift/adjoint views
 *
 * \c shift() / \c adj() only update metadata (no full-volume copies).
 * Values are resolved by \ref LatticeColorMatrix::load_at:
 *   - local / non-split dims: EO index arithmetic into SOA/SOA12 (zero copy)
 *   - MPI domain boundary: read from face/edge halo buffers only
 *     (halo buffers remain full SOA; SOA12 is single-process for now)
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
                                      int mu,
                                      ArrayType atype = ArrayType::SOA) {
    LatticeColorMatrix f;
    f.data_ = base;
    f.stride_ = stride;
    f.link_dir_ = mu;
    f.atype_ = atype;
    return f;
  }

  KOKKOS_INLINE_FUNCTION const ComplexT *data() const { return data_; }
  KOKKOS_INLINE_FUNCTION int64_t stride() const { return stride_; }
  KOKKOS_INLINE_FUNCTION int link_dir() const { return link_dir_; }
  KOKKOS_INLINE_FUNCTION ArrayType array_type() const { return atype_; }
  KOKKOS_INLINE_FUNCTION bool adjoint() const { return adjoint_; }

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
   * No field copies. Interior / serial: one EO→coords decode, apply the whole
   * shift chain on coordinates, one coords→EO encode, then SOA/SOA12 load.
   * MPI: only sites whose shift chain crosses a split-domain face use \p halo.
   */
  KOKKOS_INLINE_FUNCTION
  void load_at(int64_t idx_eo, const LatticeParams &p,
               const GaugeHaloDevice<Real> *halo, MatrixT &U) const {
    // Address resolution only; the Nc^2 element copy (and the adjoint, folded
    // into that copy) happens exactly once at the end. Each branch below is
    // scalar index arithmetic, so device code size stays O(Nc^2), not
    // O(branches * Nc^2).
    loadGaugeLinkRef(resolve_at(idx_eo, p, halo), adjoint_, U);
  }

  /// Locate the matrix this view evaluates to at EO site \p idx_eo.
  KOKKOS_INLINE_FUNCTION
  GaugeLinkRef<Real> resolve_at(int64_t idx_eo, const LatticeParams &p,
                                const GaugeHaloDevice<Real> *halo) const {
    // No shifts: the evaluation site itself.
    if (n_shifts_ == 0) {
      return gaugeLinkRefSoa(data_, idx_eo, link_dir_, stride_, p, atype_);
    }

    // Fast path: every shift direction is local (no MPI split).
    bool local_eo = true;
    if (p.mpi) {
      for (int s = 0; s < n_shifts_; ++s) {
        if (p.proc_grid[shift_mu_[s]] > 1) {
          local_eo = false;
          break;
        }
      }
    }

    // Decode once, then apply the full shift chain in coordinate space.
    const int oddbit = (idx_eo >= p.half_volume) ? 1 : 0;
    const int64_t id = idx_eo - static_cast<int64_t>(oddbit) * p.half_volume;
    int x[NDIMS];
    eo_to_coords(id, oddbit, x, p);
    for (int s = 0; s < n_shifts_; ++s) {
      x[shift_mu_[s]] += shift_sign_[s];
    }

    if (local_eo) {
      // Periodic wrap (all dims here are non-split).
      for (int d = 0; d < NDIMS; ++d) {
        const int g = p.grid[d];
        int v = x[d] % g;
        if (v < 0) {
          v += g;
        }
        x[d] = v;
      }
      const int64_t idx = coords_to_eo_idx(x, p);
      return gaugeLinkRefSoa(data_, idx, link_dir_, stride_, p, atype_);
    }
    // May leave the subdomain; halo path handles out-of-range coords.
    // Halo buffers are always full SOA; local interior uses atype_.
    return resolveGaugeLinkAtCoords(data_, stride_, halo, x, link_dir_, p,
                                    atype_);
  }

  /**
   * @brief \ref resolve_at for a site whose coordinates \p x0 are already
   *        decoded from \p idx_eo.
   *
   * Used by cross-site SIMD batches, which decode each lane once and then
   * resolve every staple leg from those coordinates. Shift chains move a
   * coordinate by at most \c LCM_MAX_SHIFTS, so the periodic wrap is done by
   * add/subtract instead of integer division.
   */
  KOKKOS_INLINE_FUNCTION
  GaugeLinkRef<Real> resolve_at_coords(int64_t idx_eo, const int x0[NDIMS],
                                       const LatticeParams &p,
                                       const GaugeHaloDevice<Real> *halo) const {
    if (n_shifts_ == 0) {
      return gaugeLinkRefSoa(data_, idx_eo, link_dir_, stride_, p, atype_);
    }

    bool local_eo = true;
    if (p.mpi) {
      for (int s = 0; s < n_shifts_; ++s) {
        if (p.proc_grid[shift_mu_[s]] > 1) {
          local_eo = false;
          break;
        }
      }
    }

    int x[NDIMS];
    for (int d = 0; d < NDIMS; ++d) {
      x[d] = x0[d];
    }
    for (int s = 0; s < n_shifts_; ++s) {
      x[shift_mu_[s]] += shift_sign_[s];
    }

    if (local_eo) {
      for (int d = 0; d < NDIMS; ++d) {
        const int g = p.grid[d];
        while (x[d] < 0) {
          x[d] += g;
        }
        while (x[d] >= g) {
          x[d] -= g;
        }
      }
      const int64_t idx = coords_to_eo_idx(x, p);
      return gaugeLinkRefSoa(data_, idx, link_dir_, stride_, p, atype_);
    }
    return resolveGaugeLinkAtCoords(data_, stride_, halo, x, link_dir_, p,
                                    atype_);
  }

private:
  const ComplexT *data_{nullptr};
  int64_t stride_{0};
  int link_dir_{0};
  ArrayType atype_{ArrayType::SOA};
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
  LatticeGaugeLinks(const ComplexT *data, int64_t soa_stride,
                    ArrayType atype = ArrayType::SOA)
      : data_(data), stride_(soa_stride), atype_(atype) {}

  KOKKOS_INLINE_FUNCTION const ComplexT *data() const { return data_; }
  KOKKOS_INLINE_FUNCTION int64_t stride() const { return stride_; }
  KOKKOS_INLINE_FUNCTION ArrayType array_type() const { return atype_; }

  KOKKOS_INLINE_FUNCTION
  MatrixT operator[](int mu) const {
    return MatrixT::gauge_soa(data_, stride_, mu, atype_);
  }

private:
  const ComplexT *data_{nullptr};
  int64_t stride_{0};
  ArrayType atype_{ArrayType::SOA};
};

/// Lazy QDP-style shift: \c result(x) = field(x + dir * e_mu).  No data move.
template <typename Real>
KOKKOS_INLINE_FUNCTION LatticeColorMatrix<Real>
shift(const LatticeColorMatrix<Real> &field, ShiftDirection dir, int mu) {
  return field.with_shift(dir, mu);
}

} // namespace kwqft

#endif
