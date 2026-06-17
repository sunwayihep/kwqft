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
#include "gauge_halo.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"
#include "neighbor_access.hpp"
#include "mpi_layout.hpp"

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif

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
  GaugeHaloBuffers<Real> *halo_;
  Real plaqValue_;
  Real spatialValue_;
  Real temporalValue_;
  double time_;

public:
  Plaquette(GaugeT &gauge, const LatticeParams &params,
            GaugeHaloBuffers<Real> *halo = nullptr)
      : gauge_(gauge), params_(params), halo_(halo), plaqValue_(0),
        spatialValue_(0), temporalValue_(0), time_(0) {}

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

    if (halo_ && params.mpi) {
      halo_->exchange(gaugeView.data(), size, params);
    }
    const GaugeHaloDevice<Real> halo_dev =
        (halo_ && params.mpi) ? halo_->device_view() : GaugeHaloDevice<Real>{};

    Real plaqSum = 0;
    Real spatialSum = 0;
    Real temporalSum = 0;

    Kokkos::parallel_reduce(
        "Plaquette", Kokkos::RangePolicy<DefaultExecSpace>(0, volume),
        KOKKOS_LAMBDA(const int64_t idd, Real &lsum, Real &ssum, Real &tsum) {
          (void)lsum;
          ComplexT *gaugePtr = gaugeView.data();

          int oddbit = 0;
          int64_t id = idd;
          if (idd >= halfVol) {
            oddbit = 1;
            id = idd - halfVol;
          }

          int x[NDIMS];
          eo_to_coords(id, oddbit, x, params);

          for (int mu = 0; mu < NDIMS; ++mu) {
            MatrixT uMuX;
            loadGaugeLinkAtCoords(gaugePtr, size, halo_dev, x, mu, params,
                                  uMuX);
            int xpmu[NDIMS];
            for (int d = 0; d < NDIMS; ++d) {
              xpmu[d] = x[d];
            }
            xpmu[mu]++;

            for (int nu = mu + 1; nu < NDIMS; ++nu) {
              MatrixT uNuXmu, uMuXnu, uNuX;
              loadGaugeLinkAtCoords(gaugePtr, size, halo_dev, xpmu, nu, params,
                                    uNuXmu);
              int xpnu[NDIMS];
              for (int d = 0; d < NDIMS; ++d) {
                xpnu[d] = x[d];
              }
              xpnu[nu]++;
              loadGaugeLinkAtCoords(gaugePtr, size, halo_dev, xpnu, mu, params,
                                    uMuXnu);
              loadGaugeLinkAtCoords(gaugePtr, size, halo_dev, x, nu, params,
                                    uNuX);
              MatrixT link = uNuXmu * uMuXnu.dagger() * uNuX.dagger();
              Real tr = (uMuX * link).realtrace();

              if (nu == NDIMS - 1) {
                tsum += tr;
              } else {
                ssum += tr;
              }
            }
          }
        },
        plaqSum, spatialSum, temporalSum);

#ifdef KWQFT_USE_MPI
    if (params_.mpi) {
      double loc[2] = {static_cast<double>(spatialSum),
                       static_cast<double>(temporalSum)};
      double glob[2];
      MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      spatialSum = static_cast<Real>(glob[0]);
      temporalSum = static_cast<Real>(glob[1]);
    }
#endif

    int64_t norm_vol = params_.volume;
    if (params_.mpi) {
      norm_vol = 1;
      for (int d = 0; d < NDIMS; ++d) {
        norm_vol *= static_cast<int64_t>(params_.global_grid[d]);
      }
    }

    spatialValue_ =
        spatialSum / (Real(NCOLORS) * norm_vol * TOTAL_NUM_SPLAQS);
    temporalValue_ =
        temporalSum / (Real(NCOLORS) * norm_vol * TOTAL_NUM_TPLAQS);
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
    if (params_.mpi && mpi_comm_rank() != 0) {
      return;
    }
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

    if (params_.mpi && params_.proc_grid[NDIMS - 1] != 1) {
      polyValue_ = ComplexT(0, 0);
      time_ = timer.seconds();
      return;
    }

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

#ifdef KWQFT_USE_MPI
    if (params_.mpi && params_.proc_grid[NDIMS - 1] == 1) {
      double lr[2] = {static_cast<double>(polyRe), static_cast<double>(polyIm)};
      double gr[2];
      MPI_Allreduce(lr, gr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      int64_t gsp = 1;
      for (int i = 0; i < NDIMS - 1; ++i) {
        gsp *= static_cast<int64_t>(params_.global_grid[i]);
      }
      polyValue_ = ComplexT(static_cast<Real>(gr[0] / static_cast<double>(gsp)),
                            static_cast<Real>(gr[1] / static_cast<double>(gsp)));
    } else
#endif
    {
      polyValue_ =
          ComplexT(polyRe / spatialVolume, polyIm / spatialVolume);
    }

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
    if (params_.mpi && mpi_comm_rank() != 0) {
      return;
    }
    if (params_.mpi && params_.proc_grid[NDIMS - 1] != 1) {
      printf("Polyakov Loop: N/A (MPI split along time direction)\n");
      return;
    }
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
