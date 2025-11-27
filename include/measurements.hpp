/**
 * @file measurements.hpp
 * @brief Measurement observables for KWQFT
 *
 * Implements plaquette and Polyakov loop measurements
 * using Kokkos parallel reductions
 */

#ifndef KWQFT_MEASUREMENTS_HPP
#define KWQFT_MEASUREMENTS_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "gauge_array.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"

namespace kwqft {

//=============================================================================
// Plaquette measurement
//=============================================================================

/**
 * @brief Calculate plaquette expectation value
 *
 * Computes the average plaquette = (1/Nc) * Re Tr(U_plaq)
 * where U_plaq = U_mu(x) * U_nu(x+mu) * U_mu^\dagger(x+nu) * U_nu^\dagger(x)
 *
 * Uses even-odd (checkerboard) storage format
 */
template <typename Real> class Plaquette {
public:
  using GaugeT = GaugeArray<Real>;
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;

private:
  GaugeT &gauge_;
  LatticeParams params_;
  Real plaqValue_;
  Real spatialValue_;
  Real temporalValue_;
  double time_;

  /**
   * @brief Helper function to load a gauge link from even-odd storage
   */
  KOKKOS_INLINE_FUNCTION
  static void loadLink(MatrixT &U, const ComplexT *gaugePtr, int64_t idx,
                       int mu, int64_t volume, int64_t size) {
    int64_t muvolume = mu * volume;
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        U.e[i][j] = gaugePtr[idx + muvolume + (j + i * NCOLORS) * size];
      }
    }
  }

  /**
   * @brief Get neighbor index in even-odd ordering (simplified for single GPU)
   */
  KOKKOS_INLINE_FUNCTION
  static int64_t getNeighborEo(int64_t id, int oddbit, int mu, int lmu,
                               const LatticeParams &p) {
    // Convert even-odd index to coordinates using indexNdEo logic
    int x[NDIMS];
    int64_t factor = id / (p.grid[0] / 2);
    for (int i = 1; i < NDIMS; ++i) {
      int64_t factor1 = factor / p.grid[i];
      x[i] = static_cast<int>(factor - factor1 * p.grid[i]);
      factor = factor1;
    }
    int sum = 0;
    for (int i = 1; i < NDIMS; ++i) {
      sum += x[i];
    }
    int xodd = (sum + oddbit) & 1;
    x[0] = static_cast<int>((id * 2 + xodd) - id / (p.grid[0] / 2) * p.grid[0]);

    // Move to neighbor
    x[mu] = (x[mu] + lmu + p.grid[mu]) % p.grid[mu];

    // Convert coordinates to normal linear index, then to EO index
    int64_t pos = 0;
    factor = 1;
    for (int i = 0; i < NDIMS; ++i) {
      pos += x[i] * factor;
      factor *= p.grid[i];
    }
    pos /= 2; // Convert to half-volume index

    // Determine parity of neighbor and add offset for odd sites
    int sumX = 0;
    for (int i = 0; i < NDIMS; ++i) {
      sumX += x[i];
    }
    int oddbit1 = sumX & 1;
    pos += oddbit1 * p.half_volume;

    return pos;
  }

public:
  Plaquette(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params), plaqValue_(0), spatialValue_(0),
        temporalValue_(0), time_(0) {}

  /**
   * @brief Compute the plaquette using even-odd storage format
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    auto params = params_;
    int64_t size = gauge_.size();
    int64_t volume = params.volume;
    int64_t halfVol = params.half_volume;

    Real plaqSum = 0;
    Real spatialSum = 0;
    Real temporalSum = 0;

    // Parallel reduction over all sites (even + odd)
    Kokkos::parallel_reduce(
        "Plaquette", Kokkos::RangePolicy<DefaultExecSpace>(0, volume),
        KOKKOS_LAMBDA(const int64_t idd, Real &lsum, Real &ssum, Real &tsum) {
          ComplexT *gaugePtr = gaugeView.data();

          // Determine parity and half-volume index
          int oddbit = 0;
          int64_t id = idd;
          if (idd >= halfVol) {
            oddbit = 1;
            id = idd - halfVol;
          }

          // Current site index in even-odd storage
          int64_t idxoddbit = id + oddbit * halfVol;

          // Calculate plaquettes for all mu < nu directions
          MatrixT link1, link;

          for (int mu = 0; mu < NDIMS; ++mu) {
            // Load U_mu(x)
            loadLink(link1, gaugePtr, idxoddbit, mu, volume, size);

            // Get neighbor index x + mu
            int64_t newidmu1 = getNeighborEo(id, oddbit, mu, 1, params);

            for (int nu = mu + 1; nu < NDIMS; ++nu) {
              // Load U_nu(x+mu)
              MatrixT uNuXmu;
              loadLink(uNuXmu, gaugePtr, newidmu1, nu, volume, size);

              // Load U_mu(x+nu) - dagger
              int64_t newidnu1 = getNeighborEo(id, oddbit, nu, 1, params);
              MatrixT uMuXnu;
              loadLink(uMuXnu, gaugePtr, newidnu1, mu, volume, size);

              // Load U_nu(x) - dagger
              MatrixT uNu;
              loadLink(uNu, gaugePtr, idxoddbit, nu, volume, size);

              // Compute U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
              link = uNuXmu * uMuXnu.dagger() * uNu.dagger();
              Real tr = (link1 * link).realtrace();

              // Separate spatial and temporal plaquettes
              if (nu == NDIMS - 1) {
                tsum += tr; // Temporal plaquette (mu,T) or (T,nu)
              } else {
                ssum += tr; // Spatial plaquette
              }
            }
          }
        },
        plaqSum, spatialSum, temporalSum);

    // Normalize: divide by Nc and number of plaquettes
    // Average over different spatial and time directions
    spatialValue_ = spatialSum / (Real(NCOLORS) * volume * TOTAL_NUM_SPLAQS);
    temporalValue_ = temporalSum / (Real(NCOLORS) * volume * TOTAL_NUM_TPLAQS);
    plaqValue_ = (spatialValue_ + temporalValue_) / Real(2);

    time_ = timer.seconds();
  }

  Real value() const { return plaqValue_; }
  Real spatial() const { return spatialValue_; }
  Real temporal() const { return temporalValue_; }
  double time() const { return time_; }

  /**
   * @brief Calculate number of floating point operations
   *
   * The factor 120 includes all plaquette directions per site
   */
  long long flop() const {
    return static_cast<long long>(NCOLORS) * NCOLORS * NCOLORS * 120LL *
           params_.volume;
  }

  /**
   * @brief Calculate bytes read
   *
   */
  long long bytes() const {
    int numParams = NCOLORS * NCOLORS * 2; // SOA format (real + imag)
    return (22LL * numParams + 4LL) * params_.volume * sizeof(Real);
  }

  double flops() const {
    return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
  }

  double bandwidth() const {
    return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30))
                       : 0.0;
  }

  void printValue() const {
    printf("Plaquette: %.12f (spatial: %.12f, temporal: %.12f)\n",
           static_cast<double>(plaqValue_), static_cast<double>(spatialValue_),
           static_cast<double>(temporalValue_));
  }

  void stat() const {
    printf("Plaquette:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", time_, bandwidth(),
           flops());
  }
};

//=============================================================================
// Polyakov loop measurement
//=============================================================================

/**
 * @brief Calculate Polyakov loop
 *
 * The Polyakov loop is the trace of the product of temporal links
 * at a fixed spatial position
 */
template <typename Real> class PolyakovLoop {
public:
  using GaugeT = GaugeArray<Real>;
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;

private:
  GaugeT &gauge_;
  LatticeParams params_;
  ComplexT polyValue_;
  double time_;

public:
  PolyakovLoop(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params), polyValue_(0, 0), time_(0) {}

  /**
   * @brief Compute the Polyakov loop
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    auto params = params_;
    int64_t size = gauge_.size();

    // Calculate spatial volume
    int64_t spatialVolume = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      spatialVolume *= params.grid[i];
    }

    int nt = params.grid[NDIMS - 1];
    int64_t volume = params.volume;
    int tDir = NDIMS - 1;
    int64_t tVolume = tDir * volume;

    Real polyRe = 0;
    Real polyIm = 0;

    // Parallel reduction over spatial sites
    Kokkos::parallel_reduce(
        "PolyakovLoop",
        Kokkos::RangePolicy<DefaultExecSpace>(0, spatialVolume),
        KOKKOS_LAMBDA(const int64_t spatialIdx, Real &reSum, Real &imSum) {
          ComplexT *gaugePtr = gaugeView.data();

          // Convert spatial index to coordinates
          int x[NDIMS];
          int64_t temp = spatialIdx;
          for (int i = 0; i < NDIMS - 1; ++i) {
            x[i] = static_cast<int>(temp % params.grid[i]);
            temp /= params.grid[i];
          }
          x[NDIMS - 1] = 0; // Start at t=0

          // Product of temporal links
          MatrixT poly = MatrixT::identity();

          for (int t = 0; t < nt; ++t) {
            x[NDIMS - 1] = t;
            int64_t idx = indexNdNm<NDIMS>(x, params);

            // Load temporal link
            MatrixT uT;
            for (int i = 0; i < NCOLORS; ++i) {
              for (int j = 0; j < NCOLORS; ++j) {
                uT.e[i][j] =
                    gaugePtr[idx + tVolume + (j + i * NCOLORS) * size];
              }
            }

            poly = poly * uT;
          }

          // Take trace
          ComplexT tr = poly.trace() / Real(NCOLORS);
          reSum += tr.real();
          imSum += tr.imag();
        },
        polyRe, polyIm);

    // Average over spatial volume
    polyValue_ = ComplexT(polyRe / spatialVolume, polyIm / spatialVolume);

    time_ = timer.seconds();
  }

  ComplexT value() const { return polyValue_; }
  Real absValue() const { return polyValue_.abs(); }
  double time() const { return time_; }

  /**
   * @brief Calculate number of floating point operations
   */
  long long flop() const {
    int nt = params_.grid[NDIMS - 1];
    long long spatialVolume = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      spatialVolume *= params_.grid[i];
    }
#if (NCOLORS == 3)
    return (4LL + 198LL * nt) * spatialVolume;
#else
    return ((NCOLORS - 1) * 2LL +
            static_cast<long long>(NCOLORS) * NCOLORS * NCOLORS * 8LL * nt) *
           spatialVolume;
#endif
  }

  /**
   * @brief Calculate bytes read
   */
  long long bytes() const {
    int nt = params_.grid[NDIMS - 1];
    long long spatialVolume = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      spatialVolume *= params_.grid[i];
    }
    int numParams = NCOLORS * NCOLORS * 2;
    return spatialVolume * (numParams * nt + 2LL) * sizeof(Real);
  }

  double flops() const {
    return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
  }

  double bandwidth() const {
    return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30))
                       : 0.0;
  }

  void printValue() const {
    printf("Polyakov Loop: %.12f + %.12f i (|P| = %.12f)\n",
           static_cast<double>(polyValue_.real()),
           static_cast<double>(polyValue_.imag()),
           static_cast<double>(absValue()));
  }

  void stat() const {
    printf("Polyakov Loop:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", time_,
           bandwidth(), flops());
  }
};

//=============================================================================
// Reunitarization
//=============================================================================

/**
 * @brief Reunitarize gauge field to enforce SU(N) constraint
 *
 * Uses Gram-Schmidt orthonormalization
 */
template <typename Real> class Reunitarize {
public:
  using GaugeT = GaugeArray<Real>;
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ComplexT = Complex<Real>;

private:
  GaugeT &gauge_;
  LatticeParams params_;
  double time_;

  /**
   * @brief Gram-Schmidt reunitarization for a single matrix
   */
  KOKKOS_INLINE_FUNCTION
  static void reunitarizeMatrix(MatrixT &U) {
    if constexpr (NCOLORS == 3) {
      // For SU(3), use the simplified method
      // Normalize first row
      Real norm = Real(0);
      for (int j = 0; j < 3; ++j) {
        norm += U.e[0][j].abs2();
      }
      norm = Real(1) / Kokkos::sqrt(norm);
      for (int j = 0; j < 3; ++j) {
        U.e[0][j] *= norm;
      }

      // Orthogonalize second row to first
      ComplexT dot = ComplexT::zero();
      for (int j = 0; j < 3; ++j) {
        dot += ~U.e[0][j] * U.e[1][j];
      }
      for (int j = 0; j < 3; ++j) {
        U.e[1][j] -= dot * U.e[0][j];
      }

      // Normalize second row
      norm = Real(0);
      for (int j = 0; j < 3; ++j) {
        norm += U.e[1][j].abs2();
      }
      norm = Real(1) / Kokkos::sqrt(norm);
      for (int j = 0; j < 3; ++j) {
        U.e[1][j] *= norm;
      }

      // Third row is cross product of first two
      U.e[2][0] = ~(U.e[0][1] * U.e[1][2] - U.e[0][2] * U.e[1][1]);
      U.e[2][1] = ~(U.e[0][2] * U.e[1][0] - U.e[0][0] * U.e[1][2]);
      U.e[2][2] = ~(U.e[0][0] * U.e[1][1] - U.e[0][1] * U.e[1][0]);
    } else {
      // General Gram-Schmidt for SU(N)
      for (int row = 0; row < NCOLORS; ++row) {
        // Orthogonalize against previous rows
        for (int prev = 0; prev < row; ++prev) {
          ComplexT dot = ComplexT::zero();
          for (int j = 0; j < NCOLORS; ++j) {
            dot += ~U.e[prev][j] * U.e[row][j];
          }
          for (int j = 0; j < NCOLORS; ++j) {
            U.e[row][j] -= dot * U.e[prev][j];
          }
        }

        // Normalize
        Real norm = Real(0);
        for (int j = 0; j < NCOLORS; ++j) {
          norm += U.e[row][j].abs2();
        }
        norm = Real(1) / Kokkos::sqrt(norm);
        for (int j = 0; j < NCOLORS; ++j) {
          U.e[row][j] *= norm;
        }
      }
    }
  }

public:
  Reunitarize(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params), time_(0) {}

  /**
   * @brief Reunitarize all links
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    int64_t size = gauge_.size();      // volume * NDIMS
    int64_t totalLinks = params_.size; // volume * NDIMS

    Kokkos::parallel_for(
        "Reunitarize", Kokkos::RangePolicy<DefaultExecSpace>(0, totalLinks),
        KOKKOS_LAMBDA(const int64_t linkIdx) {
          ComplexT *gaugePtr = gaugeView.data();

          // Load matrix from SOA format
          // Index: linkIdx + elemIdx * size
          MatrixT U;
          for (int i = 0; i < NCOLORS; ++i) {
            for (int j = 0; j < NCOLORS; ++j) {
              U.e[i][j] = gaugePtr[linkIdx + (j + i * NCOLORS) * size];
            }
          }

          // Reunitarize
          reunitarizeMatrix(U);

          // Store back
          for (int i = 0; i < NCOLORS; ++i) {
            for (int j = 0; j < NCOLORS; ++j) {
              gaugePtr[linkIdx + (j + i * NCOLORS) * size] = U.e[i][j];
            }
          }
        });
    Kokkos::fence();

    time_ = timer.seconds();
  }

  double time() const { return time_; }

  /**
   * @brief Calculate number of floating point operations
   */
  long long flop() const {
#if (NCOLORS == 3)
    // For SOA format, getNumFlop returns 0, so just use 126LL for reunit ops
    long long flopPerLink = 126LL;
#else
    // General Gram-Schmidt complexity
    unsigned int tmpGs = 0;
    unsigned int tmpDet = 0;
    for (int i = 0; i < NCOLORS; i++) {
      tmpGs += i + 1;
      tmpDet += i;
    }
    tmpDet = tmpGs * NCOLORS * 8 + tmpDet * (NCOLORS * 8 + 11);
    tmpGs = tmpGs * NCOLORS * 16 + NCOLORS * (NCOLORS * 6 + 2);
    long long flopPerLink = static_cast<long long>(tmpGs + tmpDet);
#endif
    return flopPerLink * params_.size;
  }

  /**
   * @brief Calculate bytes read/written
   */
  long long bytes() const {
    int numParams = NCOLORS * NCOLORS * 2;
    // Read + write one matrix per link
    return 2LL * numParams * sizeof(Real) * params_.size;
  }

  double flops() const {
    return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
  }

  double bandwidth() const {
    return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30))
                       : 0.0;
  }

  void stat() const {
    printf("Reunitarize:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", time_, bandwidth(),
           flops());
  }
};

} // namespace kwqft

#endif // KWQFT_MEASUREMENTS_HPP
