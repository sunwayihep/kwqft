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
#include "gauge_ops.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"
#include "neighbor_access.hpp"
#include "perf_stats.hpp"
#include "shift.hpp"
#include "lattice_color_matrix_algebra.hpp"
#include "mpi_layout.hpp"

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif
#include <memory>
#include <vector>

namespace kwqft {

//=============================================================================
// Plaquette measurement
//=============================================================================

/**
 * @brief Calculate plaquette expectation value
 *
 * Formula style (lazy shift/adj, one reduction kernel per plane):
 *   Tr( U_mu * shift(U_nu,+mu) * adj(shift(U_mu,+nu)) * adj(U_nu) )
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

public:
  Plaquette(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params), plaqValue_(0), spatialValue_(0),
        temporalValue_(0), time_(0) {}

  /**
   * @brief Compute the plaquette.
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    int64_t size = gauge_.size();

    // Shared halo: only blocks left stale by the last update are exchanged.
    GaugeHaloBuffers<Real> *halo = gauge_.halo(params_);
    if (halo) {
      halo->refresh(gaugeView.data(), size);
    }
    const GaugeHaloDevice<Real> halo_dev =
        halo ? halo->device_view() : GaugeHaloDevice<Real>{};
    const GaugeHaloDevice<Real> *halo_ptr = halo ? &halo_dev : nullptr;

    Real plaqSum = 0;
    Real spatialSum = 0;
    Real temporalSum = 0;

    const LatticeGaugeLinks<Real> u(gaugeView.data(), size, gauge_.type());

    for (int mu = 1; mu < NDIMS; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        Real pairSum = realTraceSum(
            u[mu] * shift(u[nu], FORWARD, mu) *
                adj(shift(u[mu], FORWARD, nu)) * adj(u[nu]),
            "Plaquette", halo_ptr);

        plaqSum += pairSum;
        if (mu == t_dir() || nu == t_dir()) {
          temporalSum += pairSum;
        } else {
          spatialSum += pairSum;
        }
      }
    }

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

    const Real inv_nc_vol = Real(1) / (Real(NCOLORS) * Real(norm_vol));
    // NDIMS=2: TOTAL_NUM_SPLAQS=0 (only one plaquette plane, counted as temporal).
    if constexpr (TOTAL_NUM_SPLAQS > 0) {
      spatialValue_ = spatialSum * inv_nc_vol / Real(TOTAL_NUM_SPLAQS);
    } else {
      spatialValue_ = Real(0);
    }
    temporalValue_ = temporalSum * inv_nc_vol / Real(TOTAL_NUM_TPLAQS);
    if constexpr (TOTAL_NUM_SPLAQS > 0) {
      // Equal weight of spatial/temporal averages (matches anisotropic reporting).
      plaqValue_ = (spatialValue_ + temporalValue_) / Real(2);
    } else {
      plaqValue_ = temporalValue_;
    }

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
    int numParams = gauge_num_params(gauge_.type());
    return (22LL * numParams + 4LL) * params_.volume * sizeof(Real);
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

  void printValue() const {
    if (params_.mpi && mpi_comm_rank() != 0) {
      return;
    }
    printf("Plaquette: %.12f (spatial: %.12f, temporal: %.12f)\n",
           static_cast<double>(plaqValue_), static_cast<double>(spatialValue_),
           static_cast<double>(temporalValue_));
  }

  void stat() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    if (mpi_comm_rank() != 0) {
      return;
    }
    printf("Plaquette:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", report.time,
           report.bandwidth_gbs, report.gflops);
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
  using PolyView = Kokkos::View<MatrixT *, DefaultMemSpace>;
  using PolyHostView = typename PolyView::host_mirror_type;

private:
  using PinnedView = Kokkos::View<MatrixT *, Kokkos::SharedHostPinnedSpace>;

  GaugeT &gauge_;
  LatticeParams params_;
  ComplexT polyValue_;
  double time_;
  PolyView local_poly_;  // per-rank partial products (device)
  PolyView recv_poly_;   // partner's partial products (device)
  PinnedView host_send_; // staging when MPI cannot read device memory
  PinnedView host_recv_;
  int64_t poly_spatial_vol_{0};
#ifdef KWQFT_USE_MPI
  MPI_Comm t_comm_{MPI_COMM_NULL}; // ranks sharing my spatial block, ordered in t
#endif

  void ensure_mpi_workspace(int64_t spatialVolume) {
    if (poly_spatial_vol_ == spatialVolume) {
      return;
    }
    local_poly_ = PolyView(
        Kokkos::view_alloc("PolyakovLoop_local", Kokkos::WithoutInitializing),
        spatialVolume);
    recv_poly_ = PolyView(
        Kokkos::view_alloc("PolyakovLoop_recv", Kokkos::WithoutInitializing),
        spatialVolume);
    const int64_t n_stage =
        kwqft_mpi_uses_device_buffers() ? 0 : spatialVolume;
    host_send_ = PinnedView(
        Kokkos::view_alloc("PolyakovLoop_hsend", Kokkos::WithoutInitializing),
        n_stage);
    host_recv_ = PinnedView(
        Kokkos::view_alloc("PolyakovLoop_hrecv", Kokkos::WithoutInitializing),
        n_stage);
    poly_spatial_vol_ = spatialVolume;
  }

#ifdef KWQFT_USE_MPI
  MPI_Comm t_comm() {
    if (t_comm_ == MPI_COMM_NULL) {
      int remain[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        remain[d] = (d == NDIMS - 1) ? 1 : 0;
      }
      MPI_Cart_sub(kwqft_mpi_cart_comm(), remain, &t_comm_);
    }
    return t_comm_;
  }
#endif

public:
  PolyakovLoop(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params), polyValue_(0, 0), time_(0) {}

  ~PolyakovLoop() {
#ifdef KWQFT_USE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && t_comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&t_comm_);
    }
#endif
  }

  PolyakovLoop(const PolyakovLoop &) = delete;
  PolyakovLoop &operator=(const PolyakovLoop &) = delete;

  /**
   * @brief Compute the Polyakov loop
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    auto params = params_;
    int64_t size = gauge_.size();
    const ArrayType atype = gauge_.type();

    int64_t spatialVolume = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      spatialVolume *= params.grid[i];
    }

    const int nt = params.grid[NDIMS - 1];
    const int tDir = NDIMS - 1;

#ifdef KWQFT_USE_MPI
    const int t_nproc = params.mpi ? params.proc_grid[tDir] : 1;
    const int t_coord = params.mpi ? params.coord[tDir] : 0;
    const bool mpi_time_split = params.mpi && t_nproc > 1;
#else
    const int t_nproc = 1;
    const int t_coord = 0;
    const bool mpi_time_split = false;
#endif

    Real polyRe = 0;
    Real polyIm = 0;

    if (mpi_time_split) {
      // Each rank forms the product of its local temporal links (device).
      // The t-column then combines them by a binary tree: at stride s, rank
      // t (t % 2s == 0) receives from t+s and forms P[t..t+2s) = P_t * P_{t+s}
      // on the device. log2(t_nproc) messages per rank instead of a serial
      // chain; t_coord 0 ends with the full loop and reduces the trace.
      ensure_mpi_workspace(spatialVolume);
      auto local_poly = local_poly_;

      Kokkos::parallel_for(
          "PolyakovLoop_local",
          range_policy(0, spatialVolume),
          KOKKOS_LAMBDA(const int64_t spatialIdx) {
            ComplexT *gaugePtr = gaugeView.data();

            int x[NDIMS];
            int64_t temp = spatialIdx;
            for (int i = 0; i < NDIMS - 1; ++i) {
              x[i] = static_cast<int>(temp % params.grid[i]);
              temp /= params.grid[i];
            }

            MatrixT poly = MatrixT::identity();
            for (int t = 0; t < nt; ++t) {
              x[tDir] = t;
              const int64_t idx_eo = coords_to_eo_idx(x, params);
              MatrixT uT;
              loadGaugeLinkSoa(gaugePtr, idx_eo, tDir, size, params, uT, atype);
              poly = poly * uT;
            }
            local_poly(spatialIdx) = poly;
          });
      Kokkos::fence();

      bool holds_result = true;
#ifdef KWQFT_USE_MPI
      constexpr bool dev_mpi = kwqft_mpi_uses_device_buffers();
      const int nbytes =
          static_cast<int>(spatialVolume * static_cast<int64_t>(sizeof(MatrixT)));
      MPI_Comm tc = t_comm();
      auto recv_poly = recv_poly_;
      for (int stride = 1; stride < t_nproc; stride *= 2) {
        const int rel = t_coord % (2 * stride);
        if (rel == 0) {
          const int partner = t_coord + stride;
          if (partner >= t_nproc) {
            continue; // no partner at this level (non power of two)
          }
          void *rbuf = dev_mpi ? static_cast<void *>(recv_poly_.data())
                               : static_cast<void *>(host_recv_.data());
          MPI_Recv(rbuf, nbytes, MPI_BYTE, partner, 8100 + stride, tc,
                   MPI_STATUS_IGNORE);
          if (!dev_mpi) {
            Kokkos::deep_copy(recv_poly_, host_recv_);
          }
          Kokkos::parallel_for(
              "PolyakovLoop_combine", range_policy(0, spatialVolume),
              KOKKOS_LAMBDA(const int64_t s) {
                local_poly(s) = local_poly(s) * recv_poly(s);
              });
          Kokkos::fence();
        } else if (rel == stride) {
          const int partner = t_coord - stride;
          const void *sbuf = static_cast<const void *>(local_poly_.data());
          if (!dev_mpi) {
            Kokkos::deep_copy(host_send_, local_poly_);
            sbuf = static_cast<const void *>(host_send_.data());
          }
          MPI_Send(sbuf, nbytes, MPI_BYTE, partner, 8100 + stride, tc);
          holds_result = false;
          break;
        }
      }
#endif

      if (holds_result) {
        Kokkos::parallel_reduce(
            "PolyakovLoop_trace", range_policy(0, spatialVolume),
            KOKKOS_LAMBDA(const int64_t s, Real &reSum, Real &imSum) {
              const ComplexT tr = local_poly(s).trace() / Real(NCOLORS);
              reSum += tr.real();
              imSum += tr.imag();
            },
            polyRe, polyIm);
      }
    } else {
      Kokkos::parallel_reduce(
          "PolyakovLoop",
          range_policy(0, spatialVolume),
          KOKKOS_LAMBDA(const int64_t spatialIdx, Real &reSum, Real &imSum) {
            ComplexT *gaugePtr = gaugeView.data();

            int x[NDIMS];
            int64_t temp = spatialIdx;
            for (int i = 0; i < NDIMS - 1; ++i) {
              x[i] = static_cast<int>(temp % params.grid[i]);
              temp /= params.grid[i];
            }
            x[tDir] = 0;

            MatrixT poly = MatrixT::identity();
            for (int t = 0; t < nt; ++t) {
              x[tDir] = t;
              const int64_t idx_eo = coords_to_eo_idx(x, params);
              MatrixT uT;
              loadGaugeLinkSoa(gaugePtr, idx_eo, tDir, size, params, uT, atype);
              poly = poly * uT;
            }

            const ComplexT tr = poly.trace() / Real(NCOLORS);
            reSum += tr.real();
            imSum += tr.imag();
          },
          polyRe, polyIm);
    }

#ifdef KWQFT_USE_MPI
    if (params_.mpi) {
      double lr[2] = {static_cast<double>(polyRe), static_cast<double>(polyIm)};
      double gr[2];
      MPI_Allreduce(lr, gr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      int64_t global_spatial = 1;
      for (int i = 0; i < NDIMS - 1; ++i) {
        global_spatial *= static_cast<int64_t>(params_.global_grid[i]);
      }
      polyValue_ =
          ComplexT(static_cast<Real>(gr[0] / static_cast<double>(global_spatial)),
                   static_cast<Real>(gr[1] / static_cast<double>(global_spatial)));
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
    int numParams = gauge_num_params(gauge_.type());
    return spatialVolume * (numParams * nt + 2LL) * sizeof(Real);
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

  void printValue() const {
    if (params_.mpi && mpi_comm_rank() != 0) {
      return;
    }
    printf("Polyakov Loop: %.12f + %.12f i (|P| = %.12f)\n",
           static_cast<double>(polyValue_.real()),
           static_cast<double>(polyValue_.imag()),
           static_cast<double>(absValue()));
  }

  void stat() const {
    const auto report =
        make_perf_report(flop(), bytes(), time_, params_.mpi, params_.nproc);
    if (mpi_comm_rank() != 0) {
      return;
    }
    printf("Polyakov Loop:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", report.time,
           report.bandwidth_gbs, report.gflops);
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

public:
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

  /// Functor form of \ref reunitarizeMatrix (for ghost buffers).
  struct MatrixFunctor {
    KOKKOS_INLINE_FUNCTION void operator()(MatrixT &U) const {
      reunitarizeMatrix(U);
    }
  };

  Reunitarize(GaugeT &gauge, const LatticeParams &params)
      : gauge_(gauge), params_(params), time_(0) {}

  /**
   * @brief Reunitarize all links
   *
   * MPI: the same deterministic link-wise map is applied to the ghost copies,
   * so the shared halo stays valid and no re-exchange is needed afterwards.
   */
  void run() {
    Kokkos::Timer timer;

    auto gaugeView = gauge_.getView();
    int64_t size = gauge_.size();      // volume * NDIMS
    int64_t totalLinks = params_.size; // volume * NDIMS
    const ArrayType atype = gauge_.type();

    Kokkos::parallel_for(
        "Reunitarize", range_policy(0, totalLinks),
        KOKKOS_LAMBDA(const int64_t linkIdx) {
          ComplexT *gaugePtr = gaugeView.data();

          MatrixT U;
          loadGaugeMatrix(gaugePtr, linkIdx, size, atype, U);
          reunitarizeMatrix(U);
          storeGaugeMatrix(gaugePtr, linkIdx, size, atype, U);
        });
    Kokkos::fence();

    if (GaugeHaloBuffers<Real> *halo = gauge_.halo(params_)) {
      if (halo->all_valid()) {
        halo->apply_to_ghosts(MatrixFunctor{});
      }
    }

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
    int numParams = gauge_num_params(gauge_.type());
    // Read + write one matrix per link
    return 2LL * numParams * sizeof(Real) * params_.size;
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
    printf("Reunitarize:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", report.time,
           report.bandwidth_gbs, report.gflops);
  }
};

} // namespace kwqft

#endif // KWQFT_MEASUREMENTS_HPP
