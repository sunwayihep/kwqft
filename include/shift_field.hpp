/**
 * @file shift_field.hpp
 * @brief QDPXX-style lattice field types for \c shift(field, dir, mu)
 *
 * Naming follows QDP++:
 *   - \ref LatticeColorMatrix  — one SU(N) link field (QDP \c u[mu] or \c shift result)
 *   - \ref LatticeGaugeLinks   — all \c Nd directions (QDP \c multi1d<LatticeColorMatrix>)
 */

#ifndef KWQFT_SHIFT_FIELD_HPP
#define KWQFT_SHIFT_FIELD_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "kwqft_common.hpp"
#include "shift_layout.hpp"

namespace kwqft {

constexpr int gauge_matrix_elems() { return NCOLORS * NCOLORS; }

/// One SU(N) matrix per lattice site (QDP \c LatticeColorMatrix).
///
/// Either a view into one SOA link direction \c u[mu], or a dense EO buffer
/// returned by \c shift — same logical type as in QDP++.
template <typename Real> class LatticeColorMatrix {
public:
  using value_type = Real;
  using ComplexT = Complex<Real>;
  static constexpr int site_elems = gauge_matrix_elems();

  LatticeColorMatrix() = default;

  /// View of link direction \p mu in SOA gauge storage (QDP \c u[mu]).
  KOKKOS_INLINE_FUNCTION
  static LatticeColorMatrix gauge_soa(const ComplexT *base, int64_t stride,
                                      int mu) {
    LatticeColorMatrix f;
    f.data_ = base;
    f.stride_ = stride;
    f.link_dir_ = mu;
    f.storage_ = Storage::GaugeSoa;
    return f;
  }

  /// Dense EO field produced by \c shift.
  static LatticeColorMatrix shifted(const ComplexT *dense) {
    LatticeColorMatrix f;
    f.data_ = dense;
    f.storage_ = Storage::DenseEo;
    return f;
  }

  KOKKOS_INLINE_FUNCTION
  const ComplexT *data() const { return data_; }
  KOKKOS_INLINE_FUNCTION
  bool is_gauge_soa() const { return storage_ == Storage::GaugeSoa; }
  KOKKOS_INLINE_FUNCTION
  bool is_dense() const { return storage_ == Storage::DenseEo; }
  KOKKOS_INLINE_FUNCTION
  int64_t stride() const { return stride_; }
  KOKKOS_INLINE_FUNCTION
  int link_dir() const { return link_dir_; }

private:
  enum class Storage { GaugeSoa, DenseEo };
  const ComplexT *data_{nullptr};
  int64_t stride_{0};
  int link_dir_{0};
  Storage storage_{Storage::DenseEo};
};

/// All \c Nd link directions (QDP \c multi1d<LatticeColorMatrix> \c u).
template <typename Real> class LatticeGaugeLinks {
public:
  using value_type = Real;
  using ComplexT = Complex<Real>;
  using MatrixT = LatticeColorMatrix<Real>;

  LatticeGaugeLinks() = default;
  LatticeGaugeLinks(const ComplexT *data, int64_t soa_stride)
      : data_(data), stride_(soa_stride) {}

  KOKKOS_INLINE_FUNCTION
  const ComplexT *data() const { return data_; }
  KOKKOS_INLINE_FUNCTION
  int64_t stride() const { return stride_; }

  KOKKOS_INLINE_FUNCTION
  MatrixT operator[](int mu) const {
    return MatrixT::gauge_soa(data_, stride_, mu);
  }

private:
  const ComplexT *data_{nullptr};
  int64_t stride_{0};
};

template <typename Real>
ShiftLayout<Real> shift_layout_of(const LatticeColorMatrix<Real> &field,
                                  int64_t volume) {
  if (field.is_gauge_soa()) {
    return ShiftLayout<Real>::gauge_soa(field.stride(), field.link_dir(), volume);
  }
  return ShiftLayout<Real>::dense(LatticeColorMatrix<Real>::site_elems);
}

} // namespace kwqft

#endif
