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
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"
#include "msu2.hpp"
#include "random.hpp"

namespace kwqft {

//=============================================================================
// Device functions for staple calculation
//=============================================================================

/**
 * @brief Calculate staple at a given site and direction
 *
 * The staple is the sum of products of links that form plaquettes
 * with the link at (id, mu)
 */
template <typename Real>
KOKKOS_INLINE_FUNCTION MatrixSun<Real, NCOLORS>
calculateStaple(const Complex<Real> *gauge_ptr, int id, int oddbit, int mu,
                int size, const LatticeParams &params) {
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;

  MatrixT staple = MatrixT::zero();

  int mustride = params.volume;
  int muvolume = mu * mustride;
  int offset = params.size;

  // Index of current site in even/odd layout
  int idxoddbit = id + oddbit * params.half_volume;

  // Get neighbor in mu direction
  int newidmu1 = indexNdNeigEo(id, oddbit, mu, 1, params);

  for (int nu = 0; nu < NDIMS; ++nu) {
    if (mu == nu)
      continue;

    int nuvolume = nu * mustride;
    MatrixT link;

    // Upper staple: U_nu(x) * U_mu(x+nu) * U_nu^\dagger(x+mu)
    // Load U_nu(x)
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        link.e[i][j] =
            gauge_ptr[idxoddbit + nuvolume + (j + i * NCOLORS) * size];
      }
    }

    // Multiply by U_mu(x+nu)
    int newidnu1 = indexNdNeigEo(id, oddbit, nu, 1, params);
    MatrixT tmp1;
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        tmp1.e[i][j] =
            gauge_ptr[newidnu1 + muvolume + (j + i * NCOLORS) * size];
      }
    }
    link = link * tmp1;

    // Multiply by U_nu^\dagger(x+mu)
    MatrixT tmp2;
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        tmp2.e[i][j] =
            gauge_ptr[newidmu1 + nuvolume + (j + i * NCOLORS) * size];
      }
    }
    link = link * tmp2.dagger();

    staple += link;

    // Lower staple: U_nu^\dagger(x-nu) * U_mu(x-nu) * U_nu(x+mu-nu)
    int newidnum1 = indexNdNeigEo(id, oddbit, nu, -1, params);

    // Load U_nu^\dagger(x-nu)
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        link.e[i][j] =
            gauge_ptr[newidnum1 + nuvolume + (j + i * NCOLORS) * size];
      }
    }
    link = link.dagger();

    // Multiply by U_mu(x-nu)
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        tmp1.e[i][j] =
            gauge_ptr[newidnum1 + muvolume + (j + i * NCOLORS) * size];
      }
    }
    link = link * tmp1;

    // Multiply by U_nu(x+mu-nu)
    int newidmun = indexNdNeigEo(id, oddbit, mu, 1, nu, -1, params);
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        tmp2.e[i][j] =
            gauge_ptr[newidmun + nuvolume + (j + i * NCOLORS) * size];
      }
    }
    link = link * tmp2;

    staple += link;
  }

  return staple;
}

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
  double time_;
  int size_;

public:
  HeatBath(GaugeT &gauge, RandomGenerator &rng, const LatticeParams &params)
      : gauge_(gauge), rng_(rng), params_(params), time_(0.0) {
    size_ = params.half_volume;
  }

  /**
   * @brief Run one sweep of pseudo-heatbath
   */
  void run() {
    Kokkos::Timer timer;

    auto &pool = rng_.getPool();
    auto gauge_view = gauge_.getView();
    auto params = params_;
    int size = gauge_.size();
    int half_vol = params.half_volume;
    double beta_over_nc = params.beta_over_nc;

    // Loop over parities (even/odd)
    for (int parity = 0; parity < 2; ++parity) {
      // Loop over directions
      for (int mu = 0; mu < NDIMS; ++mu) {
        Kokkos::parallel_for(
            "HeatBath", range_policy(0, half_vol), KOKKOS_LAMBDA(const int id) {
              // Get random generator for this thread
              auto gen = pool.get_state();

              ComplexT *gauge_ptr = gauge_view.data();

              // Calculate staple (sum of neighboring plaquettes)
              MatrixT staple = calculateStaple<Real>(gauge_ptr, id, parity, mu,
                                                     size, params);

              // Get current link index
              int idxoddbit = id + parity * half_vol;
              int muvolume = mu * params.volume;

              // Load current link
              MatrixT U;
              for (int i = 0; i < NCOLORS; ++i) {
                for (int j = 0; j < NCOLORS; ++j) {
                  U.e[i][j] = gauge_ptr[idxoddbit + muvolume +
                                        (j + i * NCOLORS) * size];
                }
              }

              // Apply heatbath update
              heatBathSun<Real>(U, staple.dagger(), beta_over_nc, gen);

              // Store updated link
              for (int i = 0; i < NCOLORS; ++i) {
                for (int j = 0; j < NCOLORS; ++j) {
                  gauge_ptr[idxoddbit + muvolume + (j + i * NCOLORS) * size] =
                      U.e[i][j];
                }
              }

              // Return generator state to pool
              pool.free_state(gen);
            });
        Kokkos::fence();
      }
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
    // Flop count per site for heatbath
    // Staple: 2*(NDIMS-1) matrix multiplications + additions
    // Each SU(N) matrix multiply: ~8*N^3 flops
    // HeatBath update per SU(2) subgroup: ~100 flops
#if (NCOLORS == 3)
    long long stapleflop = 2268LL; // Staple calculation
    long long phbflop = 801LL;     // Pseudo-heatbath update
    long long ThreadFlop = (stapleflop + phbflop) * size_;
#else
    double phbflop =
        NCOLORS * NCOLORS * NCOLORS +
        (NCOLORS * (NCOLORS - 1) / 2) * (46LL + 48LL + 56LL * NCOLORS);
    double stapleflop = NCOLORS * NCOLORS * NCOLORS * 84LL;
    long long ThreadFlop =
        static_cast<long long>((stapleflop + phbflop) * size_);
#endif
    // Factor of 8 = 2 parities * 4 directions
    return ThreadFlop * 2 * NDIMS;
  }

  /**
   * @brief Calculate bytes read/written
   */
  long long bytes() const {
    // Read: 7 links for staple + 1 link to update + RNG state
    // Write: 1 link + RNG state
    int nuparams_ = NCOLORS * NCOLORS * 2; // SOA format
    // RNG state: XorShift64 uses 128 bytes (conservative estimate)
    long long rng_state_size = 128LL;
    long long bytes_per_site =
        (20LL * nuparams_ * sizeof(Real) + 2LL * rng_state_size);
    return bytes_per_site * size_ * 2 * NDIMS;
  }

  /**
   * @brief Get GFlops performance
   */
  double flops() const {
    return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
  }

  /**
   * @brief Get bandwidth in GB/s
   */
  double bandwidth() const {
    return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30))
                       : 0.0;
  }

  /**
   * @brief Print statistics
   */
  void stat() const {
    printf("HeatBath:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", time_, bandwidth(),
           flops());
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
   * @brief Run one sweep of overrelaxation
   */
  void run() {
    Kokkos::Timer timer;

    auto gauge_view = gauge_.getView();
    auto params = params_;
    int size = gauge_.size();
    int half_vol = params.half_volume;

    for (int parity = 0; parity < 2; ++parity) {
      for (int mu = 0; mu < NDIMS; ++mu) {
        Kokkos::parallel_for(
            "Overrelaxation", range_policy(0, half_vol),
            KOKKOS_LAMBDA(const int id) {
              ComplexT *gauge_ptr = gauge_view.data();

              MatrixT staple = calculateStaple<Real>(gauge_ptr, id, parity, mu,
                                                     size, params);

              int idxoddbit = id + parity * half_vol;
              int muvolume = mu * params.volume;

              MatrixT U;
              for (int i = 0; i < NCOLORS; ++i) {
                for (int j = 0; j < NCOLORS; ++j) {
                  U.e[i][j] = gauge_ptr[idxoddbit + muvolume +
                                        (j + i * NCOLORS) * size];
                }
              }

              overrelaxationSun<Real>(U, staple.dagger());

              for (int i = 0; i < NCOLORS; ++i) {
                for (int j = 0; j < NCOLORS; ++j) {
                  gauge_ptr[idxoddbit + muvolume + (j + i * NCOLORS) * size] =
                      U.e[i][j];
                }
              }
            });
        Kokkos::fence();
      }
    }

    time_ = timer.seconds();
  }

  double time() const { return time_; }

  /**
   * @brief Calculate number of floating point operations
   */
  long long flop() const {
    // Staple + overrelaxation update
#if (NCOLORS == 3)
    long long stapleflop = 2268LL;
    long long ovrflop = 801LL; // Similar to heatbath without RNG
    long long ThreadFlop = (stapleflop + ovrflop) * params_.half_volume;
#else
    double ovrflop =
        NCOLORS * NCOLORS * NCOLORS +
        (NCOLORS * (NCOLORS - 1) / 2) * (46LL + 48LL + 56LL * NCOLORS);
    double stapleflop = NCOLORS * NCOLORS * NCOLORS * 84LL;
    long long ThreadFlop =
        static_cast<long long>((stapleflop + ovrflop) * params_.half_volume);
#endif
    return ThreadFlop * 2 * NDIMS;
  }

  /**
   * @brief Calculate bytes read/written
   */
  long long bytes() const {
    int nuparams_ = NCOLORS * NCOLORS * 2;
    long long bytes_per_site = 20LL * nuparams_ * sizeof(Real);
    return bytes_per_site * params_.half_volume * 2 * NDIMS;
  }

  double flops() const {
    return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
  }

  double bandwidth() const {
    return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30))
                       : 0.0;
  }

  void stat() const {
    printf("Overrelaxation:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", time_,
           bandwidth(), flops());
  }
};

} // namespace kwqft

#endif // KWQFT_MONTE_HPP
