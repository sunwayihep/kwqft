/**
 * @file shift_map.hpp
 * @brief QDPXX-style generic lattice shift with MPI nearest-neighbor exchange
 *
 * Matches QDP \c ArrayBiDirectionalMap / global \c shift:
 *   dest(x) = src(x + isign * e_mu)
 *
 * \c shift() is layout-driven and works for any per-site dof.
 */

#ifndef KWQFT_SHIFT_MAP_HPP
#define KWQFT_SHIFT_MAP_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "shift.hpp"
#include "shift_field.hpp"
#include "shift_layout.hpp"
#include "kwqft_common.hpp"
#include <vector>

namespace kwqft {

/// Max dense shift fields alive between \ref ShiftMap::begin_sweep calls.
constexpr int SHIFT_BUF_COUNT = 5 * (NDIMS > 1 ? NDIMS - 1 : 1);

/// Tangential face volume for direction \p mu (product of grid[d], d != mu).
KOKKOS_INLINE_FUNCTION int64_t shift_face_volume(int mu, const LatticeParams &p) {
  int64_t vol = 1;
  for (int d = 0; d < NDIMS; ++d) {
    if (d != mu) {
      vol *= static_cast<int64_t>(p.grid[d]);
    }
  }
  return vol;
}

/**
 * @brief QDPXX-style shift map (cf. global \c QDP::shift).
 *
 * Returns a dense even/odd buffer: \c result(x) = field(x + isign * e_mu).
 */
template <typename Real> class ShiftMap {
public:
  using ComplexT = Complex<Real>;
  using Layout = ShiftLayout<Real>;

  explicit ShiftMap(const LatticeParams &p);

  const LatticeParams &params() const { return p_; }
  int64_t volume() const { return p_.volume; }

  /// Reset internal buffer pool (call at start of each sub-step sweep).
  void begin_sweep() { buf_next_ = 0; }

  /// Allocate one dense EO buffer from the internal pool (for \c shift / \c evaluate).
  ComplexT *allocate_dense(int site_elems) { return alloc_dense_buf(site_elems); }

  /// Generic shift for any registered \ref ShiftLayout.
  const ComplexT *shift(const ComplexT *field, const Layout &layout, int isign,
                        int mu);

  const ComplexT *shift(const ComplexT *field, const Layout &layout,
                        ShiftDirection dir, int mu) {
    return shift(field, layout, static_cast<int>(dir), mu);
  }

  /// CUDA requires lambdas in public member functions (NVCC extended-lambda rule).
  void shift_impl(const ComplexT *src, ComplexT *dst, const Layout &layout,
                  int shift_mu, int isign);

private:
  LatticeParams p_{};
  int64_t face_vol_[NDIMS]{};
  int cached_site_elems_{0};
  int buf_next_{0};

  Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>
      d_dense_[SHIFT_BUF_COUNT];
  std::vector<Kokkos::View<ComplexT *, Kokkos::HostSpace>> h_send_;
  std::vector<Kokkos::View<ComplexT *, Kokkos::HostSpace>> h_recv_;
  std::vector<Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>>
      d_ghost_;

  void prepare_site_elems(int site_elems);
  ComplexT *alloc_dense_buf(int site_elems);
};

/// Global shift map (QDPXX-style; initialized by \ref initializeParams).
template <typename Real> ShiftMap<Real> &shiftMap();

/// Create / destroy global map (called from \ref initializeParams / \ref finalizeParams).
template <typename Real> void initializeShiftMap(const LatticeParams &p);
template <typename Real> void finalizeShiftMap();

/// Reset shift buffer pool at the start of a sub-step sweep.
template <typename Real> void beginShiftSweep() { shiftMap<Real>().begin_sweep(); }

/// QDP-style: \c result(x) = field(x + dir * e_mu).
template <typename Real>
inline LatticeColorMatrix<Real> shift(const LatticeColorMatrix<Real> &field,
                                      ShiftDirection dir, int mu) {
  auto &m = shiftMap<Real>();
  return LatticeColorMatrix<Real>::shifted(
      m.shift(field.data(), shift_layout_of<Real>(field, m.volume()), dir, mu));
}

/// Low-level shift with explicit layout (extensions / custom fields).
template <typename Real>
inline const Complex<Real> *shift(const Complex<Real> *field,
                                  const ShiftLayout<Real> &layout,
                                  ShiftDirection dir, int mu) {
  return shiftMap<Real>().shift(field, layout, dir, mu);
}

} // namespace kwqft

#endif
