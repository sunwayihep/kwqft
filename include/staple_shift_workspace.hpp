/**
 * @file staple_shift_workspace.hpp
 * @brief Batched shift buffers for staple (MPI-oriented two-phase path)
 *
 * Phase 1: pack each direction to dense [volume][Nc²], then apply forward shifts
 * shift(U[ν], FORWARD, μ), shift(U[μ], FORWARD, ν) for all μ≠ν. Call
 * \ref halo_exchange_gauge_soa_before_shift before shifts when using MPI.
 * Phase 2: \ref calculateStapleTwoPhase reads upper staple from cache; lower
 * staple uses on-demand SOA loads (same as \ref calculateStaple).
 */

#ifndef KWQFT_STAPLE_SHIFT_WORKSPACE_HPP
#define KWQFT_STAPLE_SHIFT_WORKSPACE_HPP

#include <cstdio>

#include "complex.hpp"
#include "constants.hpp"
#include "halo_exchange.hpp"
#include "matrixsun.hpp"
#include "shift.hpp"
#include "kwqft_common.hpp"

namespace kwqft {

template <typename Real> struct StapleShiftCachePointers {
  using ComplexT = Complex<Real>;
  const ComplexT *dense[NDIMS];
  /// shift(U[ν], FORWARD, μ) → U_ν(x+e_μ) at x; use ν≠μ only
  const ComplexT *fnu_fmu[NDIMS][NDIMS];
  /// shift(U[μ], FORWARD, ν) → U_μ(x+e_ν) at x; use μ≠ν only
  const ComplexT *fmu_fnv[NDIMS][NDIMS];
  int64_t mat_elems;
};

template <typename Real>
KOKKOS_INLINE_FUNCTION void
load_dense_matrix(const Complex<Real> *dense, int64_t idx_eo, int64_t mat_elems,
                  MatrixSun<Real, NCOLORS> &U) {
  const int64_t b = idx_eo * mat_elems;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      U.e[i][j] = dense[b + j + i * NCOLORS];
    }
  }
}

/**
 * @brief Staple using upper plaquette legs from batched forward-shift cache.
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION MatrixSun<Real, NCOLORS>
calculateStapleTwoPhase(const Complex<Real> *gaugePtr, int64_t soa_stride,
                        const StapleShiftCachePointers<Real> &cache, int64_t id,
                        int oddbit, int mu, const LatticeParams &params) {
  using MatrixT = MatrixSun<Real, NCOLORS>;

  MatrixT staple = MatrixT::zero();
  const int64_t idx_eo = id + oddbit * params.half_volume;
  const int64_t me = cache.mat_elems;

  MatrixT u_nu_x, u_mu_x_plus_nu, u_nu_x_plus_mu, tmp;

  for (int nu = 0; nu < NDIMS; ++nu) {
    if (nu == mu)
      continue;

    const Real coeff = static_cast<Real>(params.coeffs[mu][nu]);

    load_dense_matrix(cache.dense[nu], idx_eo, me, u_nu_x);
    load_dense_matrix(cache.fmu_fnv[mu][nu], idx_eo, me, u_mu_x_plus_nu);
    load_dense_matrix(cache.fnu_fmu[nu][mu], idx_eo, me, u_nu_x_plus_mu);
    tmp = u_nu_x * u_mu_x_plus_nu * u_nu_x_plus_mu.dagger();
    staple += tmp * coeff;

    const int64_t idx_x_minus_nu =
        shift_eo(idx_eo, nu, SHIFT_BACKWARD, params);
    loadGaugeLinkSoa(gaugePtr, idx_x_minus_nu, nu, soa_stride, params, u_nu_x);
    loadGaugeLinkSoa(gaugePtr, idx_x_minus_nu, mu, soa_stride, params,
                     u_mu_x_plus_nu);
    loadGaugeLinkSoa(
        gaugePtr,
        shift_eo(shift_eo(idx_eo, mu, SHIFT_FORWARD, params), nu,
                 SHIFT_BACKWARD, params),
        nu, soa_stride, params, u_nu_x_plus_mu);
    tmp = u_nu_x.dagger() * u_mu_x_plus_nu * u_nu_x_plus_mu;
    staple += tmp * coeff;
  }

  return staple;
}

template <typename Real> class StapleShiftWorkspace {
public:
  using ComplexT = Complex<Real>;

  explicit StapleShiftWorkspace(const LatticeParams &p) : p_(p) {
    const int64_t vol = p.volume;
    mat_elems_ = static_cast<int64_t>(NCOLORS * NCOLORS);
    const size_t per = static_cast<size_t>(vol * mat_elems_);

    for (int d = 0; d < NDIMS; ++d) {
      char label[40];
      snprintf(label, sizeof(label), "staple_dense_%d", d);
      dense_[d] = Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>(
          Kokkos::view_alloc(label, Kokkos::WithoutInitializing), per);
    }
    for (int nu = 0; nu < NDIMS; ++nu) {
      for (int mu = 0; mu < NDIMS; ++mu) {
        if (nu == mu)
          continue;
        char l1[48], l2[48];
        snprintf(l1, sizeof(l1), "staple_fnufmu_%d_%d", nu, mu);
        snprintf(l2, sizeof(l2), "staple_fmufnv_%d_%d", mu, nu);
        fnu_fmu_[nu][mu] =
            Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>(
                Kokkos::view_alloc(l1, Kokkos::WithoutInitializing), per);
        fmu_fnv_[mu][nu] =
            Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>(
                Kokkos::view_alloc(l2, Kokkos::WithoutInitializing), per);
      }
    }
  }

  void rebuild(ComplexT *gauge_soa, int64_t soa_stride, const LatticeParams &p) {
    halo_exchange_gauge_soa_before_shift(gauge_soa, soa_stride, p);

    const LatticeParams params = p;
    ComplexT *g = gauge_soa;
    const int64_t vol = params.volume;
    const int64_t me = mat_elems_;

    for (int d = 0; d < NDIMS; ++d) {
      auto dense = dense_[d];
      const int dir = d;
      Kokkos::parallel_for(
          "pack_soa_to_dense",
          Kokkos::RangePolicy<DefaultExecSpace>(0, vol),
          KOKKOS_LAMBDA(const int64_t idx_eo) {
            const int64_t gbase =
                idx_eo + static_cast<int64_t>(dir) * params.volume;
            const int64_t tbase = idx_eo * me;
            for (int i = 0; i < NCOLORS; ++i) {
              for (int j = 0; j < NCOLORS; ++j) {
                const int ij = j + i * NCOLORS;
                dense(tbase + ij) =
                    g[gbase + static_cast<int64_t>(ij) * soa_stride];
              }
            }
          });
      Kokkos::fence();
    }

    for (int nu = 0; nu < NDIMS; ++nu) {
      for (int mu = 0; mu < NDIMS; ++mu) {
        if (nu == mu)
          continue;
        shift_link_field_forward_eo(dense_[nu], fnu_fmu_[nu][mu], mu, params);
      }
    }
    for (int mu = 0; mu < NDIMS; ++mu) {
      for (int nu = 0; nu < NDIMS; ++nu) {
        if (mu == nu)
          continue;
        shift_link_field_forward_eo(dense_[mu], fmu_fnv_[mu][nu], nu, params);
      }
    }
  }

  StapleShiftCachePointers<Real> cache_pointers() const {
    StapleShiftCachePointers<Real> c;
    c.mat_elems = mat_elems_;
    for (int d = 0; d < NDIMS; ++d) {
      c.dense[d] = dense_[d].data();
    }
    for (int nu = 0; nu < NDIMS; ++nu) {
      for (int mu = 0; mu < NDIMS; ++mu) {
        c.fnu_fmu[nu][mu] =
            (nu != mu) ? fnu_fmu_[nu][mu].data() : nullptr;
        c.fmu_fnv[mu][nu] =
            (mu != nu) ? fmu_fnv_[mu][nu].data() : nullptr;
      }
    }
    return c;
  }

private:
  LatticeParams p_;
  int64_t mat_elems_{0};
  Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace> dense_[NDIMS];
  Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>
      fnu_fmu_[NDIMS][NDIMS];
  Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>
      fmu_fnv_[NDIMS][NDIMS];
};

} // namespace kwqft

#endif
