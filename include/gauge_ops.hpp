/**
 * @file gauge_ops.hpp
 * @brief QDPXX-style gauge observables via shift()
 *
 * @code
 *   LatticeGaugeLinks<Real> u(gauge_ptr, gauge_stride);
 *   tmp_0 = shift(u[nu], FORWARD, mu) * adj(shift(u[mu], FORWARD, nu));
 *   Tr( u[mu](x) * tmp_0(x) * adj(u[nu](x)) )
 * @endcode
 */

#ifndef KWQFT_GAUGE_OPS_HPP
#define KWQFT_GAUGE_OPS_HPP

#include "lattice_color_matrix_algebra.hpp"
#include "shift_map.hpp"
#include "matrixsun.hpp"
#include "shift.hpp"

namespace kwqft {

constexpr int t_dir() { return NDIMS - 1; }

template <typename Real> struct StapleShifts {
  LatticeGaugeLinks<Real> u{};
  LatticeColorMatrix<Real> U_nu_fwd_mu[NDIMS]{};
  LatticeColorMatrix<Real> U_mu_fwd_nu[NDIMS]{};
  LatticeColorMatrix<Real> U_mu_bwd_nu[NDIMS]{};
  LatticeColorMatrix<Real> U_nu_bwd_nu[NDIMS]{};
  LatticeColorMatrix<Real> U_nu_fwd_mu_bwd_nu[NDIMS]{};
};

template <typename Real>
StapleShifts<Real> make_staple_shifts(const LatticeGaugeLinks<Real> &u, int mu) {
  beginShiftSweep<Real>();
  StapleShifts<Real> sh;
  sh.u = u;
  for (int nu = 0; nu < NDIMS; ++nu) {
    if (nu == mu) {
      continue;
    }
    sh.U_nu_fwd_mu[nu] = shift(u[nu], FORWARD, mu);
    sh.U_mu_fwd_nu[nu] = shift(u[mu], FORWARD, nu);
    sh.U_mu_bwd_nu[nu] = shift(u[mu], BACKWARD, nu);
    sh.U_nu_bwd_nu[nu] = shift(u[nu], BACKWARD, nu);
    sh.U_nu_fwd_mu_bwd_nu[nu] = shift(sh.U_nu_fwd_mu[nu], BACKWARD, nu);
  }
  return sh;
}

template <typename Real>
KOKKOS_INLINE_FUNCTION MatrixSun<Real, NCOLORS>
staple_site(const StapleShifts<Real> &sh, int64_t id, int oddbit, int mu,
            const LatticeParams &params) {
  using MatrixT = MatrixSun<Real, NCOLORS>;
  const int64_t idx_eo = id + static_cast<int64_t>(oddbit) * params.half_volume;

  MatrixT staple = MatrixT::zero();
  MatrixT u_nu_x, u_mu_xpnu, u_nu_xpmu, tmp;

  for (int nu = 0; nu < NDIMS; ++nu) {
    if (nu == mu) {
      continue;
    }
    const Real coeff = static_cast<Real>(params.coeffs[mu][nu]);

    loadGaugeLinkSoa(sh.u.data(), idx_eo, nu, sh.u.stride(), params, u_nu_x);
    loadLatticeColorMatrix(sh.U_mu_fwd_nu[nu], idx_eo, params, u_mu_xpnu);
    loadLatticeColorMatrix(sh.U_nu_fwd_mu[nu], idx_eo, params, u_nu_xpmu);
    tmp = u_nu_x * u_mu_xpnu * u_nu_xpmu.dagger();
    staple += tmp * coeff;

    loadLatticeColorMatrix(sh.U_nu_bwd_nu[nu], idx_eo, params, u_nu_x);
    loadLatticeColorMatrix(sh.U_mu_bwd_nu[nu], idx_eo, params, u_mu_xpnu);
    loadLatticeColorMatrix(sh.U_nu_fwd_mu_bwd_nu[nu], idx_eo, params, u_nu_xpmu);
    tmp = u_nu_x.dagger() * u_mu_xpnu * u_nu_xpmu;
    staple += tmp * coeff;
  }

  return staple;
}

} // namespace kwqft

#endif
