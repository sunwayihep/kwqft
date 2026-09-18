/**
 * @file lattice_color_matrix_algebra.hpp
 * @brief Lazy SU(N) algebra for \ref LatticeColorMatrix
 *
 * \c shift / \c adj are zero-copy views. Products build \ref LcmProduct;
 * \c realTraceSum evaluates in one Kokkos reduction.
 *
 * @code
 *   const Real tr = realTraceSum(
 *       u[mu] * shift(u[nu], FORWARD, mu) * adj(shift(u[mu], FORWARD, nu)) *
 *       adj(u[nu]));
 * @endcode
 */

#ifndef KWQFT_LATTICE_COLOR_MATRIX_ALGEBRA_HPP
#define KWQFT_LATTICE_COLOR_MATRIX_ALGEBRA_HPP

#include "constants.hpp"
#include "matrixsun.hpp"
#include "neighbor_access.hpp"
#include "shift_field.hpp"

namespace kwqft {

template <typename Real>
KOKKOS_INLINE_FUNCTION void loadLatticeColorMatrix(
    const LatticeColorMatrix<Real> &field, int64_t idx_eo,
    const LatticeParams &p, MatrixSun<Real, NCOLORS> &U,
    const GaugeHaloDevice<Real> *halo = nullptr) {
  field.load_at(idx_eo, p, halo, U);
}

/// Hermitian conjugate view (zero-copy).
template <typename Real>
KOKKOS_INLINE_FUNCTION LatticeColorMatrix<Real>
adj(const LatticeColorMatrix<Real> &in, const char * /*label*/ = nullptr) {
  return in.with_adjoint();
}

/// Product of N lattice color-matrix views.
template <typename Real, int N> struct LcmProduct {
  static_assert(N >= 1 && N <= 8, "LcmProduct arity out of range");
  LatticeColorMatrix<Real> f[N]{};

  KOKKOS_INLINE_FUNCTION
  void eval_at(int64_t idx_eo, const LatticeParams &p,
               const GaugeHaloDevice<Real> *halo,
               MatrixSun<Real, NCOLORS> &U) const {
    f[0].load_at(idx_eo, p, halo, U);
    for (int i = 1; i < N; ++i) {
      MatrixSun<Real, NCOLORS> Ui;
      f[i].load_at(idx_eo, p, halo, Ui);
      U = U * Ui;
    }
  }
};

template <typename Real>
KOKKOS_INLINE_FUNCTION LcmProduct<Real, 2>
operator*(const LatticeColorMatrix<Real> &a,
          const LatticeColorMatrix<Real> &b) {
  LcmProduct<Real, 2> p;
  p.f[0] = a;
  p.f[1] = b;
  return p;
}

template <typename Real, int N>
KOKKOS_INLINE_FUNCTION LcmProduct<Real, N + 1>
operator*(const LcmProduct<Real, N> &a, const LatticeColorMatrix<Real> &b) {
  LcmProduct<Real, N + 1> p;
  for (int i = 0; i < N; ++i) {
    p.f[i] = a.f[i];
  }
  p.f[N] = b;
  return p;
}

template <typename Real, int N>
KOKKOS_INLINE_FUNCTION LcmProduct<Real, N + 1>
operator*(const LatticeColorMatrix<Real> &a, const LcmProduct<Real, N> &b) {
  LcmProduct<Real, N + 1> p;
  p.f[0] = a;
  for (int i = 0; i < N; ++i) {
    p.f[i + 1] = b.f[i];
  }
  return p;
}

//=============================================================================
// Reductions
//=============================================================================

template <typename Real>
Real realTraceSum(const LatticeColorMatrix<Real> &field,
                  const char *label = "realTraceSum",
                  const GaugeHaloDevice<Real> *halo = nullptr) {
  const int64_t vol = PARAMS::params.volume;
  auto dparams = get_device_params();
  const GaugeHaloDevice<Real> halo_cap =
      halo ? *halo : GaugeHaloDevice<Real>{};
  const bool have_halo = halo != nullptr;
  Real sum = 0;

  Kokkos::parallel_reduce(
      label, range_policy(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo, Real &s) {
        const LatticeParams p = dparams();
        const GaugeHaloDevice<Real> *hp = have_halo ? &halo_cap : nullptr;
        MatrixSun<Real, NCOLORS> U;
        field.load_at(idx_eo, p, hp, U);
        s += U.realtrace();
      },
      sum);
  return sum;
}

template <typename Real, int N>
Real realTraceSum(const LcmProduct<Real, N> &prod,
                  const char *label = "realTraceSum",
                  const GaugeHaloDevice<Real> *halo = nullptr) {
  const int64_t vol = PARAMS::params.volume;
  auto dparams = get_device_params();
  const GaugeHaloDevice<Real> halo_cap =
      halo ? *halo : GaugeHaloDevice<Real>{};
  const bool have_halo = halo != nullptr;
  Real sum = 0;

  Kokkos::parallel_reduce(
      label, range_policy(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo, Real &s) {
        const LatticeParams p = dparams();
        const GaugeHaloDevice<Real> *hp = have_halo ? &halo_cap : nullptr;
        MatrixSun<Real, NCOLORS> U;
        prod.eval_at(idx_eo, p, hp, U);
        s += U.realtrace();
      },
      sum);
  return sum;
}

} // namespace kwqft

#endif
