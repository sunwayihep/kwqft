/**
 * @file gauge_ops.hpp
 * @brief QDPXX-style gauge observables via lazy shift()
 *
 * Staples use \c shift() metadata only (no volume copies). Interior sites load
 * via EO indices; MPI ranks exchange face/edge halo before evaluation so
 * boundary shifts read ghost buffers.
 */

#ifndef KWQFT_GAUGE_OPS_HPP
#define KWQFT_GAUGE_OPS_HPP

#include "lattice_color_matrix_algebra.hpp"
#include "matrixsun.hpp"
#include "neighbor_access.hpp"
#include "shift.hpp"
#include "shift_field.hpp"

namespace kwqft {

constexpr int t_dir() { return NDIMS - 1; }

/**
 * @brief Wilson staple at one EO site via lazy \c shift() views.
 *
 * Builds only small per-nu shift metadata on the stack (safe on device).
 * Do not construct a full \c StapleShifts inside a GPU kernel — that can
 * exceed per-thread stack and silently skip the launch.
 *
 * Accumulates each staple leg with in-place \c *= into a single \c link
 * matrix (plus one load buffer).
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION MatrixSun<Real, NCOLORS>
calculateStapleLazy(const Complex<Real> *gaugePtr, int64_t soa_stride,
                    const GaugeHaloDevice<Real> *halo, int64_t id, int oddbit,
                    int mu, const LatticeParams &params,
                    ArrayType atype = ArrayType::SOA) {
  using MatrixT = MatrixSun<Real, NCOLORS>;
  const LatticeGaugeLinks<Real> u(gaugePtr, soa_stride, atype);
  const int64_t idx_eo =
      id + static_cast<int64_t>(oddbit) * params.half_volume;

  MatrixT staple = MatrixT::zero();
  MatrixT link, buf;

  for (int nu = 0; nu < NDIMS; ++nu) {
    if (nu == mu) {
      continue;
    }
    const Real coeff = static_cast<Real>(params.coeffs[mu][nu]);

    const LatticeColorMatrix<Real> U_mu_fwd_nu = shift(u[mu], FORWARD, nu);
    const LatticeColorMatrix<Real> U_nu_fwd_mu = shift(u[nu], FORWARD, mu);
    const LatticeColorMatrix<Real> U_nu_bwd_nu = shift(u[nu], BACKWARD, nu);
    const LatticeColorMatrix<Real> U_mu_bwd_nu = shift(u[mu], BACKWARD, nu);
    const LatticeColorMatrix<Real> U_nu_fwd_mu_bwd_nu =
        shift(U_nu_fwd_mu, BACKWARD, nu);

    // UP: U_ν(x) U_μ(x+ν) U_ν†(x+μ)
    loadLatticeColorMatrix(u[nu], idx_eo, params, link, halo);
    loadLatticeColorMatrix(U_mu_fwd_nu, idx_eo, params, buf, halo);
    link *= buf;
    loadLatticeColorMatrix(U_nu_fwd_mu, idx_eo, params, buf, halo);
    link = UUDagger(link, buf);
    if (coeff != Real(1)) {
      link *= coeff;
    }
    staple += link;

    // DOWN: U_ν†(x−ν) U_μ(x−ν) U_ν(x−ν+μ)
    loadLatticeColorMatrix(U_nu_bwd_nu, idx_eo, params, buf, halo);
    link = buf.dagger();
    loadLatticeColorMatrix(U_mu_bwd_nu, idx_eo, params, buf, halo);
    link *= buf;
    loadLatticeColorMatrix(U_nu_fwd_mu_bwd_nu, idx_eo, params, buf, halo);
    link *= buf;
    if (coeff != Real(1)) {
      link *= coeff;
    }
    staple += link;
  }

  return staple;
}

} // namespace kwqft

#endif
