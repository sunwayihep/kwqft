/**
 * @file lattice_color_matrix_algebra.hpp
 * @brief SU(N) algebra for \ref LatticeColorMatrix
 *
 * Each operation launches its own Kokkos kernel and writes a dense EO field
 * (buffer from the global \ref ShiftMap pool).
 *
 * @code
 *   beginShiftSweep<Real>();
 *   const Real tr = realTraceSum(
 *       u[mu] * shift(u[nu], FORWARD, mu) * adj(shift(u[mu], FORWARD, nu)) *
 *       adj(u[nu]));
 * @endcode
 */

#ifndef KWQFT_LATTICE_COLOR_MATRIX_ALGEBRA_HPP
#define KWQFT_LATTICE_COLOR_MATRIX_ALGEBRA_HPP

#include "constants.hpp"
#include "matrixsun.hpp"
#include "shift.hpp"
#include "shift_field.hpp"
#include "shift_map.hpp"

namespace kwqft {

template <typename Real>
KOKKOS_INLINE_FUNCTION void loadLatticeColorMatrix(
    const LatticeColorMatrix<Real> &field, int64_t idx_eo,
    const LatticeParams &p, MatrixSun<Real, NCOLORS> &U) {
  if (field.is_gauge_soa()) {
    loadGaugeLinkSoa(field.data(), idx_eo, field.link_dir(), field.stride(), p,
                     U);
  } else {
    const int me = LatticeColorMatrix<Real>::site_elems;
    const int64_t base = idx_eo * static_cast<int64_t>(me);
    const Complex<Real> *dense = field.data();
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        U.e[i][j] = dense[base + j + i * NCOLORS];
      }
    }
  }
}

template <typename Real>
KOKKOS_INLINE_FUNCTION void storeLatticeColorMatrix(
    const MatrixSun<Real, NCOLORS> &U, int64_t idx_eo, Complex<Real> *dense) {
  const int me = LatticeColorMatrix<Real>::site_elems;
  const int64_t base = idx_eo * static_cast<int64_t>(me);
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      dense[base + j + i * NCOLORS] = U.e[i][j];
    }
  }
}

/// Read one matrix element without materializing a full MatrixSun (HIP VGPR).
template <typename Real>
KOKKOS_INLINE_FUNCTION Complex<Real> loadLcmElement(
    const Complex<Real> *data, int64_t idx_eo, int i, int j, bool is_gauge_soa,
    int link_dir, int64_t soa_stride, const LatticeParams &p) {
  if (is_gauge_soa) {
    const int64_t base =
        idx_eo + static_cast<int64_t>(link_dir) * p.volume;
    return data[base + (j + i * NCOLORS) * soa_stride];
  }
  const int64_t base =
      idx_eo * static_cast<int64_t>(LatticeColorMatrix<Real>::site_elems);
  return data[base + j + i * NCOLORS];
}

template <typename Real>
LatticeColorMatrix<Real> allocate_dense_lcm() {
  auto &map = shiftMap<Real>();
  return LatticeColorMatrix<Real>::shifted(
      map.allocate_dense(LatticeColorMatrix<Real>::site_elems));
}

//=============================================================================
// Eager Kokkos field operations
//=============================================================================

/// Hermitian conjugate: \c out(x) = adj(in(x)).
///
/// Element-wise (no full-matrix temporaries): HIP/DCU otherwise hits
/// HSA_STATUS_ERROR_OUT_OF_REGISTERS for mid-size Nc (notably Nc=6), because
/// MatrixSun + dagger() return can exceed the per-lane VGPR limit while the
/// compiler still chooses private_segment_size=0.
template <typename Real>
LatticeColorMatrix<Real> adj(const LatticeColorMatrix<Real> &in,
                             const char *label = "lcm_adj") {
  LatticeColorMatrix<Real> out = allocate_dense_lcm<Real>();
  const int64_t vol = shiftMap<Real>().volume();
  const Complex<Real> *in_data = in.data();
  Complex<Real> *out_data = const_cast<Complex<Real> *>(out.data());
  const int64_t soa_stride = in.stride();
  const int link_dir = in.link_dir();
  const bool is_gauge_soa = in.is_gauge_soa();
  auto dparams = get_device_params();

  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<DefaultExecSpace>(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo) {
        const LatticeParams p = dparams();
        const int64_t out_base =
            idx_eo * static_cast<int64_t>(LatticeColorMatrix<Real>::site_elems);
        // out[i][j] = conj(in[j][i])
        for (int i = 0; i < NCOLORS; ++i) {
          for (int j = 0; j < NCOLORS; ++j) {
            const Complex<Real> c = loadLcmElement(
                in_data, idx_eo, j, i, is_gauge_soa, link_dir, soa_stride, p);
            out_data[out_base + j + i * NCOLORS] = ~c;
          }
        }
      });
  Kokkos::fence();
  return out;
}

/// Pointwise matrix multiply: \c out(x) = a(x) * b(x).
template <typename Real>
LatticeColorMatrix<Real>
multiply(const LatticeColorMatrix<Real> &a, const LatticeColorMatrix<Real> &b,
         const char *label = "lcm_mul") {
  LatticeColorMatrix<Real> out = allocate_dense_lcm<Real>();
  const int64_t vol = shiftMap<Real>().volume();
  const Complex<Real> *a_data = a.data();
  const Complex<Real> *b_data = b.data();
  Complex<Real> *out_data = const_cast<Complex<Real> *>(out.data());
  const int64_t a_stride = a.stride();
  const int64_t b_stride = b.stride();
  const int a_dir = a.link_dir();
  const int b_dir = b.link_dir();
  const bool a_soa = a.is_gauge_soa();
  const bool b_soa = b.is_gauge_soa();
  auto dparams = get_device_params();

  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<DefaultExecSpace>(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo) {
        const LatticeParams p = dparams();
        const int64_t out_base =
            idx_eo * static_cast<int64_t>(LatticeColorMatrix<Real>::site_elems);
        for (int i = 0; i < NCOLORS; ++i) {
          for (int j = 0; j < NCOLORS; ++j) {
            Complex<Real> sum = Complex<Real>::zero();
            for (int k = 0; k < NCOLORS; ++k) {
              const Complex<Real> aik = loadLcmElement(
                  a_data, idx_eo, i, k, a_soa, a_dir, a_stride, p);
              const Complex<Real> bkj = loadLcmElement(
                  b_data, idx_eo, k, j, b_soa, b_dir, b_stride, p);
              sum += aik * bkj;
            }
            out_data[out_base + j + i * NCOLORS] = sum;
          }
        }
      });
  Kokkos::fence();
  return out;
}

/// Pointwise matrix add: \c out(x) = a(x) + b(x).
template <typename Real>
LatticeColorMatrix<Real>
add(const LatticeColorMatrix<Real> &a, const LatticeColorMatrix<Real> &b,
    const char *label = "lcm_add") {
  LatticeColorMatrix<Real> out = allocate_dense_lcm<Real>();
  const int64_t vol = shiftMap<Real>().volume();
  const Complex<Real> *a_data = a.data();
  const Complex<Real> *b_data = b.data();
  Complex<Real> *out_data = const_cast<Complex<Real> *>(out.data());
  const int64_t a_stride = a.stride();
  const int64_t b_stride = b.stride();
  const int a_dir = a.link_dir();
  const int b_dir = b.link_dir();
  const bool a_soa = a.is_gauge_soa();
  const bool b_soa = b.is_gauge_soa();
  auto dparams = get_device_params();

  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<DefaultExecSpace>(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo) {
        const LatticeParams p = dparams();
        const int64_t out_base =
            idx_eo * static_cast<int64_t>(LatticeColorMatrix<Real>::site_elems);
        for (int i = 0; i < NCOLORS; ++i) {
          for (int j = 0; j < NCOLORS; ++j) {
            const Complex<Real> aa = loadLcmElement(
                a_data, idx_eo, i, j, a_soa, a_dir, a_stride, p);
            const Complex<Real> bb = loadLcmElement(
                b_data, idx_eo, i, j, b_soa, b_dir, b_stride, p);
            out_data[out_base + j + i * NCOLORS] = aa + bb;
          }
        }
      });
  Kokkos::fence();
  return out;
}

/// Pointwise scalar multiply: \c out(x) = coeff * in(x).
template <typename Real>
LatticeColorMatrix<Real> scale(const LatticeColorMatrix<Real> &in, Real coeff,
                               const char *label = "lcm_scale") {
  LatticeColorMatrix<Real> out = allocate_dense_lcm<Real>();
  const int64_t vol = shiftMap<Real>().volume();
  const Complex<Real> *in_data = in.data();
  Complex<Real> *out_data = const_cast<Complex<Real> *>(out.data());
  const int64_t soa_stride = in.stride();
  const int link_dir = in.link_dir();
  const bool is_gauge_soa = in.is_gauge_soa();
  auto dparams = get_device_params();

  Kokkos::parallel_for(
      label, Kokkos::RangePolicy<DefaultExecSpace>(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo) {
        const LatticeParams p = dparams();
        const int64_t out_base =
            idx_eo * static_cast<int64_t>(LatticeColorMatrix<Real>::site_elems);
        for (int i = 0; i < NCOLORS; ++i) {
          for (int j = 0; j < NCOLORS; ++j) {
            const Complex<Real> c = loadLcmElement(
                in_data, idx_eo, i, j, is_gauge_soa, link_dir, soa_stride, p);
            out_data[out_base + j + i * NCOLORS] = c * coeff;
          }
        }
      });
  Kokkos::fence();
  return out;
}

template <typename Real>
LatticeColorMatrix<Real> operator*(const LatticeColorMatrix<Real> &a,
                                   const LatticeColorMatrix<Real> &b) {
  return multiply(a, b);
}

template <typename Real>
LatticeColorMatrix<Real> operator+(const LatticeColorMatrix<Real> &a,
                                   const LatticeColorMatrix<Real> &b) {
  return add(a, b);
}

template <typename Real>
LatticeColorMatrix<Real> operator*(Real coeff, const LatticeColorMatrix<Real> &a) {
  return scale(a, coeff);
}

template <typename Real>
LatticeColorMatrix<Real> operator*(const LatticeColorMatrix<Real> &a, Real coeff) {
  return scale(a, coeff);
}

//=============================================================================
// Kokkos reductions
//=============================================================================

/// Sum of Re Tr(field(x)) over local lattice sites.
template <typename Real>
Real realTraceSum(const LatticeColorMatrix<Real> &field,
                  const char *label = "realTraceSum") {
  const int64_t vol = shiftMap<Real>().volume();
  const Complex<Real> *field_data = field.data();
  const int64_t soa_stride = field.stride();
  const int link_dir = field.link_dir();
  const bool is_gauge_soa = field.is_gauge_soa();
  auto dparams = get_device_params();
  Real sum = 0;

  Kokkos::parallel_reduce(
      label, Kokkos::RangePolicy<DefaultExecSpace>(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo, Real &s) {
        const LatticeParams p = dparams();
        Real tr = Real(0);
        for (int i = 0; i < NCOLORS; ++i) {
          tr += loadLcmElement(field_data, idx_eo, i, i, is_gauge_soa,
                               link_dir, soa_stride, p)
                    .real();
        }
        s += tr;
      },
      sum);
  return sum;
}

} // namespace kwqft

#endif
