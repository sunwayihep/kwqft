/**
 * @file monte.hpp
 * @brief Monte Carlo algorithms for KWQFT
 *
 * Implements pseudo-heatbath and overrelaxation algorithms
 * using Kokkos for portable parallel execution
 */

#ifndef KWQFT_MONTE_HPP
#define KWQFT_MONTE_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "gauge_array.hpp"
#include "gauge_halo.hpp"
#include "gauge_ops.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"
#include "mpi_layout.hpp"
#include "msu2.hpp"
#include "neighbor_access.hpp"
#include "perf_stats.hpp"
#include "random.hpp"
#include "shift.hpp"
#include "shift_field.hpp"

#include <memory>

namespace kwqft {

//=============================================================================
// Device staple gather — lazy shift views (see gauge_ops.hpp)
//=============================================================================

/**
 * @brief Pseudo-heatbath update for SU(N)
 *
 * Updates a link using the pseudo-heatbath algorithm
 * by iterating over SU(2) subgroups.
 *
 * \p Real may also be a SIMD pack of independent sites; \p gen is then a
 * callable returning the SU(2) heatbath matrix for a pack of couplings,
 * drawn lane by lane from each site's own stream. The scalar case calls
 * \ref generateSu2Matrix_milc directly: wrapping it in a lambda changes
 * GCC's FMA contraction and hence the scalar/GPU trajectory.
 */
template <typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION Msu2<Real> drawHeatBathSu2(Real ap, Generator &gen) {
  if constexpr (std::is_floating_point_v<Real>) {
    return generateSu2Matrix_milc<Real>(ap, gen);
  } else {
    return gen(ap);
  }
}

template <typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION void heatBathSun(MatrixSun<Real, NCOLORS> &U,
                                        const MatrixSun<Real, NCOLORS> &F,
                                        double beta_over_nc, Generator &gen) {
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;

#if (NCOLORS == 2)
  // For SU(2), direct update
  int p = 0, q = 1;
  Msu2<Real> r = getBlockSu2<Real, NCOLORS>(F, p, q);
  Real k = r.abs();
  Real ap = Real(static_cast<scalar_of_t<Real>>(beta_over_nc)) * k;
  k = Real(1) / k;
  r *= k;
  Msu2<Real> a = drawHeatBathSu2<Real>(ap, gen);
  Msu2<Real> rr = mulsu2UVDagger<Real>(a, r);
  U = MatrixSun<Real, NCOLORS>::identity();
  U.e[0][0] = ComplexT(rr.a0(), rr.a3());
  U.e[0][1] = ComplexT(rr.a2(), rr.a1());
  U.e[1][0] = ComplexT(-rr.a2(), rr.a1());
  U.e[1][1] = ComplexT(rr.a0(), -rr.a3());

#elif (NCOLORS == 3)
  // For SU(3), iterate over 3 SU(2) subgroups
  for (int block = 0; block < 3; ++block) {
    int p, q;
    IndexBlock(block, p, q);

    // Compute (U*F) block elements
    ComplexT a0 = ComplexT::zero();
    ComplexT a1 = ComplexT::zero();
    ComplexT a2 = ComplexT::zero();
    ComplexT a3 = ComplexT::zero();

    for (int j = 0; j < NCOLORS; ++j) {
      a0 += U.e[p][j] * F.e[j][p];
      a1 += U.e[p][j] * F.e[j][q];
      a2 += U.e[q][j] * F.e[j][p];
      a3 += U.e[q][j] * F.e[j][q];
    }

    Msu2<Real> r;
    r.a0() = a0.real() + a3.real();
    r.a1() = a1.imag() + a2.imag();
    r.a2() = a1.real() - a2.real();
    r.a3() = a0.imag() - a3.imag();

    Real k = r.abs();
    Real ap = Real(static_cast<scalar_of_t<Real>>(beta_over_nc)) * k;
    k = Real(1) / k;
    r *= k;

    Msu2<Real> a = drawHeatBathSu2<Real>(ap, gen);
    r = mulsu2UVDagger<Real>(a, r);

    // Update U = su2 * U
    a0 = ComplexT(r.a0(), r.a3());
    a1 = ComplexT(r.a2(), r.a1());
    a2 = ComplexT(-r.a2(), r.a1());
    a3 = ComplexT(r.a0(), -r.a3());

    for (int j = 0; j < NCOLORS; ++j) {
      ComplexT tmp0 = a0 * U.e[p][j] + a1 * U.e[q][j];
      U.e[q][j] = a2 * U.e[p][j] + a3 * U.e[q][j];
      U.e[p][j] = tmp0;
    }
  }

#else
  // General SU(N): iterate over all N(N-1)/2 subgroups
  MatrixT M = U * F;
  for (int block = 0; block < TOTAL_SUB_BLOCKS; ++block) {
    int p, q;
    IndexBlock(block, p, q);

    Msu2<Real> r = getBlockSu2<Real, NCOLORS>(M, p, q);
    Real k = r.abs();
    Real ap = Real(static_cast<scalar_of_t<Real>>(beta_over_nc)) * k;
    k = Real(1) / k;
    r *= k;

    Msu2<Real> a = drawHeatBathSu2<Real>(ap, gen);
    Msu2<Real> rr = mulsu2UVDagger<Real>(a, r);

    mulBlockSun<Real, NCOLORS>(rr, U, p, q);
    mulBlockSun<Real, NCOLORS>(rr, M, p, q);
  }
#endif
}

/**
 * @brief Overrelaxation update for SU(N)
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION void
overrelaxationSun(MatrixSun<Real, NCOLORS> &U,
                  const MatrixSun<Real, NCOLORS> &F) {
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;

#if (NCOLORS == 3)
  for (int block = 0; block < 3; ++block) {
    int p, q;
    IndexBlock(block, p, q);

    ComplexT a0 = ComplexT::zero();
    ComplexT a1 = ComplexT::zero();
    ComplexT a2 = ComplexT::zero();
    ComplexT a3 = ComplexT::zero();

    for (int j = 0; j < NCOLORS; ++j) {
      a0 += U.e[p][j] * F.e[j][p];
      a1 += U.e[p][j] * F.e[j][q];
      a2 += U.e[q][j] * F.e[j][p];
      a3 += U.e[q][j] * F.e[j][q];
    }

    Msu2<Real> r;
    r.a0() = a0.real() + a3.real();
    r.a1() = a1.imag() + a2.imag();
    r.a2() = a1.real() - a2.real();
    r.a3() = a0.imag() - a3.imag();

    // Normalize and conjugate
    r = r.conj_normalize();

    a0 = ComplexT(r.a0(), r.a3());
    a1 = ComplexT(r.a2(), r.a1());
    a2 = ComplexT(-r.a2(), r.a1());
    a3 = ComplexT(r.a0(), -r.a3());

    // Apply twice: U <- r^2 * U
    for (int j = 0; j < NCOLORS; ++j) {
      ComplexT tmp0 = a0 * U.e[p][j] + a1 * U.e[q][j];
      ComplexT tmp1 = a2 * U.e[p][j] + a3 * U.e[q][j];
      U.e[p][j] = a0 * tmp0 + a1 * tmp1;
      U.e[q][j] = a2 * tmp0 + a3 * tmp1;
    }
  }
#else
  MatrixT M = U * F;
  for (int block = 0; block < TOTAL_SUB_BLOCKS; ++block) {
    int p, q;
    IndexBlock(block, p, q);

    Msu2<Real> r = getBlockSu2<Real, NCOLORS>(M, p, q);
    r = r.conj_normalize();

    mulBlockSun<Real, NCOLORS>(r, U, p, q);
    mulBlockSun<Real, NCOLORS>(r, U, p, q);
    mulBlockSun<Real, NCOLORS>(r, M, p, q);
    mulBlockSun<Real, NCOLORS>(r, M, p, q);
  }
#endif
}

//=============================================================================
// Per-site update bodies (shared by the serial and the MPI boundary/interior
// launches)
//=============================================================================

template <typename Real, typename PoolType>
KOKKOS_INLINE_FUNCTION void
heatBathUpdateSite(Complex<Real> *gaugePtr, int64_t soa_stride,
                   const GaugeHaloDevice<Real> *halo, int64_t id, int parity,
                   int mu, const LatticeParams &params, ArrayType atype,
                   double betaOverNc, const PoolType &pool) {
  using MatrixT = MatrixSun<Real, NCOLORS>;
  auto gen = pool.get_state(static_cast<uint64_t>(id));

  MatrixT staple = calculateStapleLazy<Real>(gaugePtr, soa_stride, halo, id,
                                             parity, mu, params, atype);

  const int64_t idxoddbit = id + parity * params.half_volume;
  const int64_t link_base = idxoddbit + mu * params.volume;

  MatrixT U;
  loadGaugeMatrix(gaugePtr, link_base, soa_stride, atype, U);
  heatBathSun<Real>(U, staple.dagger(), betaOverNc, gen);
  storeGaugeMatrix(gaugePtr, link_base, soa_stride, atype, U);

  pool.free_state(gen);
}

template <typename Real>
KOKKOS_INLINE_FUNCTION void
overrelaxUpdateSite(Complex<Real> *gaugePtr, int64_t soa_stride,
                    const GaugeHaloDevice<Real> *halo, int64_t id, int parity,
                    int mu, const LatticeParams &params, ArrayType atype) {
  using MatrixT = MatrixSun<Real, NCOLORS>;
  MatrixT staple = calculateStapleLazy<Real>(gaugePtr, soa_stride, halo, id,
                                             parity, mu, params, atype);

  const int64_t idxoddbit = id + parity * params.half_volume;
  const int64_t link_base = idxoddbit + mu * params.volume;

  MatrixT U;
  loadGaugeMatrix(gaugePtr, link_base, soa_stride, atype, U);
  overrelaxationSun<Real>(U, staple.dagger());
  storeGaugeMatrix(gaugePtr, link_base, soa_stride, atype, U);
}

#ifdef KWQFT_SITE_SIMD
//=============================================================================
// Cross-site SIMD update bodies (OpenMP host): the W sites id[0..W) of one
// batch are updated together as MatrixSun<simd<Real>>. Staple, link algebra,
// load and store are SIMD; only the SU(2) heatbath draw is per lane, from the
// same per-site RNG stream and in the same order as the scalar update.
//=============================================================================

template <typename Real, typename Simd>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void
loadLinkBatch(const Complex<Real> *gaugePtr, int64_t soa_stride,
              const int64_t *id, int parity, int mu,
              const LatticeParams &params, ArrayType atype, int64_t *link_base,
              MatrixSun<Simd, NCOLORS> &U) {
  constexpr int width = static_cast<int>(Simd::size());
  GaugeLinkRef<Real> ref[width];
  for (int lane = 0; lane < width; ++lane) {
    const int64_t idxoddbit = id[lane] + parity * params.half_volume;
    link_base[lane] = idxoddbit + mu * params.volume;
    ref[lane] =
        gaugeLinkRefSoa(gaugePtr, idxoddbit, mu, soa_stride, params, atype);
  }
  loadMatrixBatch<Real, Simd>(ref, false, U);
}

template <typename Real, typename Simd, typename PoolType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void
heatBathUpdateBatch(Complex<Real> *gaugePtr, int64_t soa_stride,
                    const GaugeHaloDevice<Real> *halo, const int64_t *id,
                    int parity, int mu, const LatticeParams &params,
                    ArrayType atype, double betaOverNc, const PoolType &pool) {
  using MatrixV = MatrixSun<Simd, NCOLORS>;
  constexpr int width = static_cast<int>(Simd::size());
  const MatrixV staple = calculateStapleLazyBatch<Real, Simd>(
      gaugePtr, soa_stride, halo, id, parity, mu, params, atype);

  int64_t link_base[width];
  MatrixV U;
  loadLinkBatch<Real, Simd>(gaugePtr, soa_stride, id, parity, mu, params,
                            atype, link_base, U);
  auto sample = [&](const Simd &ap) {
    Real a[4][width];
    for (int lane = 0; lane < width; ++lane) {
      auto gen = pool.get_state(static_cast<uint64_t>(id[lane]));
      const Msu2<Real> s = generateSu2Matrix_milc<Real>(ap[lane], gen);
      pool.free_state(gen);
      for (int c = 0; c < 4; ++c) {
        a[c][lane] = s.m_a[c];
      }
    }
    constexpr auto flag = Kokkos::Experimental::simd_flag_default;
    return Msu2<Simd>(Simd(a[0], flag), Simd(a[1], flag), Simd(a[2], flag),
                      Simd(a[3], flag));
  };
  heatBathSun<Simd>(U, staple.dagger(), betaOverNc, sample);
  storeMatrixBatch<Real, Simd>(gaugePtr, link_base, soa_stride, atype, U);
}

template <typename Real, typename Simd>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void
overrelaxUpdateBatch(Complex<Real> *gaugePtr, int64_t soa_stride,
                     const GaugeHaloDevice<Real> *halo, const int64_t *id,
                     int parity, int mu, const LatticeParams &params,
                     ArrayType atype) {
  using MatrixV = MatrixSun<Simd, NCOLORS>;
  constexpr int width = static_cast<int>(Simd::size());
  const MatrixV staple = calculateStapleLazyBatch<Real, Simd>(
      gaugePtr, soa_stride, halo, id, parity, mu, params, atype);

  int64_t link_base[width];
  MatrixV U;
  loadLinkBatch<Real, Simd>(gaugePtr, soa_stride, id, parity, mu, params,
                            atype, link_base, U);
  overrelaxationSun<Simd>(U, staple.dagger());
  storeMatrixBatch<Real, Simd>(gaugePtr, link_base, soa_stride, atype, U);
}
#endif

//=============================================================================
// One device kernel per update kind.
//
// Serial (all sites, no halo), MPI boundary (site list, halo) and MPI
// interior (site list, no halo) share a single lambda: the site-id source and
// the halo pointer are uniform runtime flags. Three separately instantiated
// kernels per (algorithm, Real) triple the device code nvcc must optimise,
// which for large Nc dominates compile time.
//=============================================================================

using SiteList = Kokkos::View<int64_t *, DefaultMemSpace>;

template <typename Real, typename PoolType>
void launchHeatBathSweep(const char *label, int64_t n, const SiteList &list,
                         bool has_list, Complex<Real> *gaugePtr, int64_t size,
                         const GaugeHaloDevice<Real> &halo_dev, bool use_halo,
                         int parity, int mu, const LatticeParams &params,
                         ArrayType atype, double betaOverNc,
                         const PoolType &pool) {
#ifdef KWQFT_SITE_SIMD
  using Simd = SiteSimd<Real>;
  if constexpr (Simd::size() > 1) {
    constexpr int width = static_cast<int>(Simd::size());
    const int64_t batches = n / width;
    Kokkos::parallel_for(
        label, range_policy(0, batches), KOKKOS_LAMBDA(const int64_t batch) {
          int64_t id[width];
          const int64_t begin = batch * width;
          for (int lane = 0; lane < width; ++lane) {
            const int64_t i = begin + lane;
            id[lane] = has_list ? list(i) : i;
          }
          heatBathUpdateBatch<Real, Simd>(
              gaugePtr, size, use_halo ? &halo_dev : nullptr, id, parity, mu,
              params, atype, betaOverNc, pool);
        });

    const int64_t tail = batches * width;
    Kokkos::parallel_for(
        label, range_policy(tail, n), KOKKOS_LAMBDA(const int64_t i) {
          const int64_t id = has_list ? list(i) : i;
          heatBathUpdateSite<Real>(
              gaugePtr, size, use_halo ? &halo_dev : nullptr, id, parity, mu,
              params, atype, betaOverNc, pool);
        });
    return;
  }
#endif
  Kokkos::parallel_for(
      label, range_policy(0, n), KOKKOS_LAMBDA(const int64_t i) {
        const int64_t id = has_list ? list(i) : i;
        heatBathUpdateSite<Real>(gaugePtr, size, use_halo ? &halo_dev : nullptr,
                                 id, parity, mu, params, atype, betaOverNc,
                                 pool);
      });
}

template <typename Real>
void launchOverrelaxSweep(const char *label, int64_t n, const SiteList &list,
                          bool has_list, Complex<Real> *gaugePtr, int64_t size,
                          const GaugeHaloDevice<Real> &halo_dev, bool use_halo,
                          int parity, int mu, const LatticeParams &params,
                          ArrayType atype) {
#ifdef KWQFT_SITE_SIMD
  using Simd = SiteSimd<Real>;
  if constexpr (Simd::size() > 1) {
    constexpr int width = static_cast<int>(Simd::size());
    const int64_t batches = n / width;
    Kokkos::parallel_for(
        label, range_policy(0, batches), KOKKOS_LAMBDA(const int64_t batch) {
          int64_t id[width];
          const int64_t begin = batch * width;
          for (int lane = 0; lane < width; ++lane) {
            const int64_t i = begin + lane;
            id[lane] = has_list ? list(i) : i;
          }
          overrelaxUpdateBatch<Real, Simd>(
              gaugePtr, size, use_halo ? &halo_dev : nullptr, id, parity, mu,
              params, atype);
        });

    const int64_t tail = batches * width;
    Kokkos::parallel_for(
        label, range_policy(tail, n), KOKKOS_LAMBDA(const int64_t i) {
          const int64_t id = has_list ? list(i) : i;
          overrelaxUpdateSite<Real>(
              gaugePtr, size, use_halo ? &halo_dev : nullptr, id, parity, mu,
              params, atype);
        });
    return;
  }
#endif
  Kokkos::parallel_for(
      label, range_policy(0, n), KOKKOS_LAMBDA(const int64_t i) {
        const int64_t id = has_list ? list(i) : i;
        overrelaxUpdateSite<Real>(gaugePtr, size,
                                  use_halo ? &halo_dev : nullptr, id, parity,
                                  mu, params, atype);
      });
}

//=============================================================================
// HeatBath class
//=============================================================================

/**
 * @brief Pseudo-heatbath Monte Carlo update class
 */
template <typename Real> class HeatBath {
public:
  using GaugeT = GaugeArray<Real>;
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;
  using PoolType = typename RandomGenerator::PoolType;

private:
  GaugeT &gauge_;
  RandomGenerator &rng_;
  LatticeParams params_;
  double time_;
  int64_t size_;

public:
  HeatBath(GaugeT &gauge, RandomGenerator &rng, const LatticeParams &params)
      : gauge_(gauge), rng_(rng), params_(params), time_(0.0) {
    size_ = params.half_volume;
  }

  /**
   * @brief Run one sweep of pseudo-heatbath
   *
   * MPI: the shared halo is refreshed only if stale. For every (parity, mu)
   * the face sites are updated first, their links are sent while the interior
   * sites are updated, so communication overlaps computation. At the end of
   * the sweep every ghost block is up to date.
   */
  void run() {
    Kokkos::Timer timer;

    auto &pool = rng_.getPool();
    auto gaugeView = gauge_.getView();
    auto params = params_;
    int64_t size = gauge_.size();
    int64_t halfVol = params.half_volume;
    double betaOverNc = params.beta_over_nc;
    const ArrayType atype = gauge_.type();
    ComplexT *gaugePtr = gaugeView.data();
    GaugeHaloBuffers<Real> *halo = gauge_.halo(params_);

    if (halo == nullptr) {
      const SiteList none;
      const GaugeHaloDevice<Real> no_halo{};
      for (int parity = 0; parity < 2; ++parity) {
        for (int mu = 0; mu < NDIMS; ++mu) {
          launchHeatBathSweep<Real>("HeatBath", halfVol, none, false, gaugePtr,
                                    size, no_halo, false, parity, mu, params,
                                    atype, betaOverNc, pool);
        }
      }
      Kokkos::fence();
      time_ = timer.seconds();
      return;
    }

    halo->refresh(gaugePtr, size);
    // Captured by value; the functor takes its address device-side.
    const GaugeHaloDevice<Real> halo_dev = halo->device_view();

    for (int parity = 0; parity < 2; ++parity) {
      const SiteList &bnd = halo->boundary_sites(parity);
      const SiteList &inr = halo->interior_sites(parity);
      for (int mu = 0; mu < NDIMS; ++mu) {
        launchHeatBathSweep<Real>("HeatBath_boundary", bnd.extent(0), bnd, true,
                                  gaugePtr, size, halo_dev, true, parity, mu,
                                  params, atype, betaOverNc, pool);
        Kokkos::fence();
        // Interior sites never read ghosts; use_halo=false is required while
        // the (mu, parity) MPI is in flight and would overwrite d_recv_.
        halo->begin_exchange(gaugePtr, size, mu, parity);

        launchHeatBathSweep<Real>("HeatBath_interior", inr.extent(0), inr, true,
                                  gaugePtr, size, halo_dev, false, parity, mu,
                                  params, atype, betaOverNc, pool);
        halo->end_exchange();
      }
    }
    Kokkos::fence();

    time_ = timer.seconds();
  }

  /**
   * @brief Get time for last run
   */
  double time() const { return time_; }

  /**
   * @brief Calculate number of floating point operations
   */
  long long flop() const {
#if (NCOLORS == 3)
    long long stapleFlop = 2268LL; // Staple calculation
    long long phbFlop = 801LL;     // Pseudo-heatbath update
    long long threadFlop = (stapleFlop + phbFlop) * size_;
#else
    long long phbFlop =
        NCOLORS * NCOLORS * NCOLORS +
        (NCOLORS * (NCOLORS - 1) / 2) * (46LL + 48LL + 56LL * NCOLORS);
    long long stapleFlop =
        static_cast<long long>(NCOLORS) * NCOLORS * NCOLORS * 84LL;
    long long threadFlop = (stapleFlop + phbFlop) * size_;
#endif
    // Factor of 2*NDIMS = 2 parities * NDIMS directions
    return threadFlop * 2 * NDIMS;
  }

  /**
   * @brief Calculate bytes read/written
   */
  long long bytes() const {
    // Read: 7 links for staple + 1 link to update + RNG state
    // Write: 1 link + RNG state
    int numParams = gauge_num_params(gauge_.type());
    // RNG state size: ~48 bytes (similar to cuRNGState)
    long long rngStateSize = 48LL;
    long long bytesPerSite =
        (20LL * numParams * sizeof(Real) + 2LL * rngStateSize);
    return bytesPerSite * size_ * 2 * NDIMS;
  }

  /**
   * @brief Get GFlops performance
   */
  double flops() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    return report.gflops;
  }

  /**
   * @brief Get bandwidth in GB/s
   */
  double bandwidth() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    return report.bandwidth_gbs;
  }

  /**
   * @brief Print statistics
   */
  void stat() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    if (mpi_comm_rank() != 0) {
      return;
    }
    printf("HeatBath:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", report.time,
           report.bandwidth_gbs, report.gflops);
  }
};

//=============================================================================
// Overrelaxation class
//=============================================================================

/**
 * @brief Overrelaxation Monte Carlo update class
 */
template <typename Real> class Overrelaxation {
public:
  using GaugeT = GaugeArray<Real>;
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;

private:
  GaugeT &gauge_;
  LatticeParams params_;
  double time_;

public:
  Overrelaxation(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params), time_(0.0) {}

  /**
   * @brief Run one sweep of overrelaxation (same halo schedule as HeatBath)
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    auto params = params_;
    int64_t size = gauge_.size();
    int64_t halfVol = params.half_volume;
    const ArrayType atype = gauge_.type();
    ComplexT *gaugePtr = gaugeView.data();
    GaugeHaloBuffers<Real> *halo = gauge_.halo(params_);

    if (halo == nullptr) {
      const SiteList none;
      const GaugeHaloDevice<Real> no_halo{};
      for (int parity = 0; parity < 2; ++parity) {
        for (int mu = 0; mu < NDIMS; ++mu) {
          launchOverrelaxSweep<Real>("Overrelaxation", halfVol, none, false,
                                     gaugePtr, size, no_halo, false, parity, mu,
                                     params, atype);
        }
      }
      Kokkos::fence();
      time_ = timer.seconds();
      return;
    }

    halo->refresh(gaugePtr, size);
    const GaugeHaloDevice<Real> halo_dev = halo->device_view();

    for (int parity = 0; parity < 2; ++parity) {
      const SiteList &bnd = halo->boundary_sites(parity);
      const SiteList &inr = halo->interior_sites(parity);
      for (int mu = 0; mu < NDIMS; ++mu) {
        launchOverrelaxSweep<Real>("Overrelaxation_boundary", bnd.extent(0),
                                   bnd, true, gaugePtr, size, halo_dev, true,
                                   parity, mu, params, atype);
        Kokkos::fence();
        halo->begin_exchange(gaugePtr, size, mu, parity);

        launchOverrelaxSweep<Real>("Overrelaxation_interior", inr.extent(0),
                                   inr, true, gaugePtr, size, halo_dev, false,
                                   parity, mu, params, atype);
        halo->end_exchange();
      }
    }
    Kokkos::fence();

    time_ = timer.seconds();
  }

  double time() const { return time_; }

  /**
   * @brief Calculate number of floating point operations
   */
  long long flop() const {
#if (NCOLORS == 3)
    long long stapleFlop = 2268LL;
    long long ovrFlop = 801LL; // Similar to heatbath without RNG
    long long threadFlop = (stapleFlop + ovrFlop) * params_.half_volume;
#else
    long long ovrFlop =
        NCOLORS * NCOLORS * NCOLORS +
        (NCOLORS * (NCOLORS - 1) / 2) * (46LL + 48LL + 56LL * NCOLORS);
    long long stapleFlop =
        static_cast<long long>(NCOLORS) * NCOLORS * NCOLORS * 84LL;
    long long threadFlop = (stapleFlop + ovrFlop) * params_.half_volume;
#endif
    // Factor of 2*NDIMS = 2 parities * NDIMS directions
    return threadFlop * 2 * NDIMS;
  }

  /**
   * @brief Calculate bytes read/written
   */
  long long bytes() const {
    int numParams = gauge_num_params(gauge_.type());
    long long bytesPerSite = 20LL * numParams * sizeof(Real);
    return bytesPerSite * params_.half_volume * 2 * NDIMS;
  }

  double flops() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    return report.gflops;
  }

  double bandwidth() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    return report.bandwidth_gbs;
  }

  void stat() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    if (mpi_comm_rank() != 0) {
      return;
    }
    printf("Overrelaxation:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", report.time,
           report.bandwidth_gbs, report.gflops);
  }
};

// Explicit instantiations live in src/monte.cpp. See measurements.hpp.
extern template class HeatBath<double>;
extern template class Overrelaxation<double>;
#ifndef KOKKOS_ENABLE_HIP
extern template class HeatBath<float>;
extern template class Overrelaxation<float>;
#endif

} // namespace kwqft

#endif // KWQFT_MONTE_HPP
