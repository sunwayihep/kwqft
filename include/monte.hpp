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
#include "neighbor_access.hpp"
#include "shift.hpp"
#include "shift_field.hpp"
#include "kwqft_common.hpp"
#include "perf_stats.hpp"
#include "mpi_layout.hpp"
#include "matrixsun.hpp"
#include "msu2.hpp"
#include "random.hpp"

#include <memory>

namespace kwqft {

//=============================================================================
// Device staple gather — lazy shift views (see gauge_ops.hpp)
//=============================================================================

/**
 * @brief Pseudo-heatbath update for SU(N)
 *
 * Updates a link using the pseudo-heatbath algorithm
 * by iterating over SU(2) subgroups
 */
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
  Real ap = static_cast<Real>(beta_over_nc) * k;
  k = Real(1) / k;
  r *= k;
  Msu2<Real> a = generateSu2Matrix_milc<Real>(ap, gen);
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
    Real ap = static_cast<Real>(beta_over_nc) * k;
    k = Real(1) / k;
    r *= k;

    Msu2<Real> a = generateSu2Matrix_milc<Real>(ap, gen);
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
    Real ap = static_cast<Real>(beta_over_nc) * k;
    k = Real(1) / k;
    r *= k;

    Msu2<Real> a = generateSu2Matrix_milc<Real>(ap, gen);
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
  std::unique_ptr<GaugeHaloBuffers<Real>> halo_;
  double time_;
  int64_t size_;

public:
  HeatBath(GaugeT &gauge, RandomGenerator &rng, const LatticeParams &params)
      : gauge_(gauge), rng_(rng), params_(params),
        halo_(make_halo_if_mpi<Real>(params)), time_(0.0) {
    size_ = params.half_volume;
  }

  /**
   * @brief Run one sweep of pseudo-heatbath
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
    const bool use_halo = static_cast<bool>(halo_);

    for (int parity = 0; parity < 2; ++parity) {
      // One full halo per parity; after each mu, only re-send that direction
      // (subsequent staples need the just-updated neighbor links).
      if (use_halo) {
        halo_->exchange(gaugeView.data(), size, params);
      }
      const GaugeHaloDevice<Real> halo_dev =
          use_halo ? halo_->device_view() : GaugeHaloDevice<Real>{};

      for (int mu = 0; mu < NDIMS; ++mu) {
        Kokkos::parallel_for(
            "HeatBath", range_policy(0, halfVol),
            KOKKOS_LAMBDA(const int64_t id) {
              auto gen = pool.get_state(static_cast<uint64_t>(id));
              ComplexT *gaugePtr = gaugeView.data();

              // Capture halo_dev by value; take address inside the functor
              // (device-local), never pass a host stack pointer into the kernel.
              const GaugeHaloDevice<Real> *hp =
                  use_halo ? &halo_dev : nullptr;
              MatrixT staple = calculateStapleLazy<Real>(
                  gaugePtr, size, hp, id, parity, mu, params, atype);

              const int64_t idxoddbit = id + parity * halfVol;
              const int64_t link_base = idxoddbit + mu * params.volume;

              MatrixT U;
              loadGaugeMatrix(gaugePtr, link_base, size, atype, U);
              heatBathSun<Real>(U, staple.dagger(), betaOverNc, gen);
              storeGaugeMatrix(gaugePtr, link_base, size, atype, U);

              pool.free_state(gen);
            });
        if (use_halo) {
          Kokkos::fence();
          if (mu + 1 < NDIMS) {
            halo_->exchange_dir(gaugeView.data(), size, params, mu);
          }
        }
      }
    }
    if (!use_halo) {
      Kokkos::fence();
    }

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
    long long stapleFlop = static_cast<long long>(NCOLORS) * NCOLORS * NCOLORS * 84LL;
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
  std::unique_ptr<GaugeHaloBuffers<Real>> halo_;
  double time_;

public:
  Overrelaxation(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params),
        halo_(make_halo_if_mpi<Real>(params)), time_(0.0) {}

  /**
   * @brief Run one sweep of overrelaxation
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    auto params = params_;
    int64_t size = gauge_.size();
    int64_t halfVol = params.half_volume;
    const ArrayType atype = gauge_.type();
    const bool use_halo = static_cast<bool>(halo_);

    for (int parity = 0; parity < 2; ++parity) {
      if (use_halo) {
        halo_->exchange(gaugeView.data(), size, params);
      }
      const GaugeHaloDevice<Real> halo_dev =
          use_halo ? halo_->device_view() : GaugeHaloDevice<Real>{};

      for (int mu = 0; mu < NDIMS; ++mu) {
        Kokkos::parallel_for(
            "Overrelaxation", range_policy(0, halfVol),
            KOKKOS_LAMBDA(const int64_t id) {
              ComplexT *gaugePtr = gaugeView.data();

              // Capture halo_dev by value; take address inside the functor
              // (device-local), never pass a host stack pointer into the kernel.
              const GaugeHaloDevice<Real> *hp =
                  use_halo ? &halo_dev : nullptr;
              MatrixT staple = calculateStapleLazy<Real>(
                  gaugePtr, size, hp, id, parity, mu, params, atype);

              const int64_t idxoddbit = id + parity * halfVol;
              const int64_t link_base = idxoddbit + mu * params.volume;

              MatrixT U;
              loadGaugeMatrix(gaugePtr, link_base, size, atype, U);
              overrelaxationSun<Real>(U, staple.dagger());
              storeGaugeMatrix(gaugePtr, link_base, size, atype, U);
            });
        if (use_halo) {
          Kokkos::fence();
          if (mu + 1 < NDIMS) {
            halo_->exchange_dir(gaugeView.data(), size, params, mu);
          }
        }
      }
    }
    if (!use_halo) {
      Kokkos::fence();
    }

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
    long long stapleFlop = static_cast<long long>(NCOLORS) * NCOLORS * NCOLORS * 84LL;
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

} // namespace kwqft

#endif // KWQFT_MONTE_HPP
