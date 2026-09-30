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
#include "lattice_color_matrix_algebra.hpp"
#include "matrixsun.hpp"
#include "mpi_layout.hpp"
#include "neighbor_access.hpp"
#include "perf_stats.hpp"
#include "shift.hpp"

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
  GaugeT &gauge;
  LatticeParams params;
  Real plaq_value;
  Real spatial_value;
  Real temporal_value;
  double m_time;

public:
  Plaquette(GaugeT &gauge, const LatticeParams &params)
      : gauge(gauge), params(params), plaq_value(0), spatial_value(0),
        temporal_value(0), m_time(0) {}

  /**
   * @brief Compute the plaquette.
   */
  void run() {
    Kokkos::Timer timer;

    auto gauge_view = gauge.getView();
    int64_t size = gauge.size();

    // Shared halo: only blocks left stale by the last update are exchanged.
    GaugeHaloBuffers<Real> *halo = gauge.halo(params);
    if (halo) {
      halo->refresh(gauge_view.data(), size);
    }
    const GaugeHaloDevice<Real> halo_dev =
        halo ? halo->deviceView() : GaugeHaloDevice<Real>{};
    const GaugeHaloDevice<Real> *halo_ptr = halo ? &halo_dev : nullptr;

    Real plaq_sum = 0;
    Real spatial_sum = 0;
    Real temporal_sum = 0;

    const LatticeGaugeLinks<Real> u(gauge_view.data(), size, gauge.type());

    for (int mu = 1; mu < NDIMS; ++mu) {
      for (int nu = 0; nu < mu; ++nu) {
        Real pair_sum =
            realTraceSum(u[mu] * shift(u[nu], FORWARD, mu) *
                             adj(shift(u[mu], FORWARD, nu)) * adj(u[nu]),
                         "Plaquette", halo_ptr);

        plaq_sum += pair_sum;
        if (mu == tDir() || nu == tDir()) {
          temporal_sum += pair_sum;
        } else {
          spatial_sum += pair_sum;
        }
      }
    }

#ifdef KWQFT_USE_MPI
    if (params.mpi) {
      double loc[2] = {static_cast<double>(spatial_sum),
                       static_cast<double>(temporal_sum)};
      double glob[2];
      MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      spatial_sum = static_cast<Real>(glob[0]);
      temporal_sum = static_cast<Real>(glob[1]);
    }
#endif

    int64_t norm_vol = params.volume;
    if (params.mpi) {
      norm_vol = 1;
      for (int d = 0; d < NDIMS; ++d) {
        norm_vol *= static_cast<int64_t>(params.global_grid[d]);
      }
    }

    const Real inv_nc_vol = Real(1) / (Real(NCOLORS) * Real(norm_vol));
    // NDIMS=2: TOTAL_NUM_SPLAQS=0 (only one plaquette plane, counted as
    // temporal).
    if constexpr (TOTAL_NUM_SPLAQS > 0) {
      spatial_value = spatial_sum * inv_nc_vol / Real(TOTAL_NUM_SPLAQS);
    } else {
      spatial_value = Real(0);
    }
    temporal_value = temporal_sum * inv_nc_vol / Real(TOTAL_NUM_TPLAQS);
    if constexpr (TOTAL_NUM_SPLAQS > 0) {
      // Equal weight of spatial/temporal averages (matches anisotropic
      // reporting).
      plaq_value = (spatial_value + temporal_value) / Real(2);
    } else {
      plaq_value = temporal_value;
    }

    m_time = timer.seconds();
  }

  Real value() const { return plaq_value; }
  Real spatial() const { return spatial_value; }
  Real temporal() const { return temporal_value; }
  double time() const { return m_time; }

  /**
   * @brief Calculate number of floating point operations.
   *
   * Each of the D(D-1)/2 planes is three SU(N) products and a real trace.
   */
  long long flop() const {
    const long long n = NCOLORS;
    const long long planes =
        static_cast<long long>(NDIMS) * (NDIMS - 1) / 2;
    const long long per_plane = 3 * sunProductFlops() + (n - 1);
    return per_plane * planes * params.volume;
  }

  /**
   * @brief Calculate bytes read
   *
   * The D=4 count is (22*num_params+4) reals per site for six planes.
   * Scale that cost by the number of planes D(D-1)/2.
   */
  long long bytes() const {
    int num_params = gaugeNumParams(gauge.type());
    const long long planes =
        static_cast<long long>(NDIMS) * (NDIMS - 1) / 2;
    const long long reals4d = 22LL * num_params + 4LL;
    return reals4d * planes * params.volume *
           static_cast<long long>(sizeof(Real)) / 6;
  }

  double flops() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    return report.gflops;
  }

  double bandwidth() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    return report.bandwidth_gbs;
  }

  void printValue() const {
    if (params.mpi && mpiCommRank() != 0) {
      return;
    }
    printf("Plaquette: %.12f (spatial: %.12f, temporal: %.12f)\n",
           static_cast<double>(plaq_value), static_cast<double>(spatial_value),
           static_cast<double>(temporal_value));
  }

  void stat() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    if (mpiCommRank() != 0) {
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

  GaugeT &gauge;
  LatticeParams params;
  ComplexT poly_value;
  double m_time;
  PolyView local_poly;  // per-rank partial products (device)
  PolyView recv_poly;   // partner's partial products (device)
  PinnedView host_send; // staging when MPI cannot read device memory
  PinnedView host_recv;
  int64_t poly_spatial_vol{0};
#ifdef KWQFT_USE_MPI
  MPI_Comm t_comm{
      MPI_COMM_NULL}; // ranks sharing my spatial block, ordered in t
#endif

  void ensureMpiWorkspace(int64_t spatial_volume) {
    if (poly_spatial_vol == spatial_volume) {
      return;
    }
    local_poly = PolyView(
        Kokkos::view_alloc("PolyakovLoop_local", Kokkos::WithoutInitializing),
        spatial_volume);
    recv_poly = PolyView(
        Kokkos::view_alloc("PolyakovLoop_recv", Kokkos::WithoutInitializing),
        spatial_volume);
    const int64_t n_stage = kwqftMpiUsesDeviceBuffers() ? 0 : spatial_volume;
    host_send = PinnedView(
        Kokkos::view_alloc("PolyakovLoop_hsend", Kokkos::WithoutInitializing),
        n_stage);
    host_recv = PinnedView(
        Kokkos::view_alloc("PolyakovLoop_hrecv", Kokkos::WithoutInitializing),
        n_stage);
    poly_spatial_vol = spatial_volume;
  }

#ifdef KWQFT_USE_MPI
  MPI_Comm tComm() {
    if (t_comm == MPI_COMM_NULL) {
      int remain[NDIMS];
      for (int d = 0; d < NDIMS; ++d) {
        remain[d] = (d == NDIMS - 1) ? 1 : 0;
      }
      MPI_Cart_sub(kwqftMpiCartComm(), remain, &t_comm);
    }
    return t_comm;
  }
#endif

public:
  PolyakovLoop(GaugeT &gauge, const LatticeParams &params)
      : gauge(gauge), params(params), poly_value(0, 0), m_time(0) {}

  ~PolyakovLoop() {
#ifdef KWQFT_USE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && t_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&t_comm);
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

    auto gauge_view = gauge.getView();
    auto params = this->params;
    int64_t size = gauge.size();
    const ArrayType atype = gauge.type();

    int64_t spatial_volume = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      spatial_volume *= params.grid[i];
    }

    const int nt = params.grid[NDIMS - 1];
    const int t_dir = NDIMS - 1;

#ifdef KWQFT_USE_MPI
    const int t_nproc = params.mpi ? params.proc_grid[t_dir] : 1;
    const int t_coord = params.mpi ? params.coord[t_dir] : 0;
    const bool mpi_time_split = params.mpi && t_nproc > 1;
#else
    const bool mpi_time_split = false;
#endif

    Real poly_re = 0;
    Real poly_im = 0;

    if (mpi_time_split) {
      // Each rank forms the product of its local temporal links (device).
      // The t-column then combines them by a binary tree: at stride s, rank
      // t (t % 2s == 0) receives from t+s and forms P[t..t+2s) = P_t * P_{t+s}
      // on the device. log2(t_nproc) messages per rank instead of a serial
      // chain; t_coord 0 ends with the full loop and reduces the trace.
      ensureMpiWorkspace(spatial_volume);
      auto local_poly = this->local_poly;

      Kokkos::parallel_for(
          "PolyakovLoop_local", RangePolicy(0, spatial_volume),
          KOKKOS_LAMBDA(const int64_t spatialIdx) {
            ComplexT *gauge_ptr = gauge_view.data();

            int x[NDIMS];
            int64_t temp = spatialIdx;
            for (int i = 0; i < NDIMS - 1; ++i) {
              x[i] = static_cast<int>(temp % params.grid[i]);
              temp /= params.grid[i];
            }

            MatrixT poly = MatrixT::identity();
            for (int t = 0; t < nt; ++t) {
              x[t_dir] = t;
              const int64_t idx_eo = coordsToEoIdx(x, params);
              MatrixT uT;
              loadGaugeLinkSoa(gauge_ptr, idx_eo, t_dir, size, params, uT, atype);
              poly *= uT;
            }
            local_poly(spatialIdx) = poly;
          });
      Kokkos::fence();

      bool holds_result = true;
#ifdef KWQFT_USE_MPI
      constexpr bool dev_mpi = kwqftMpiUsesDeviceBuffers();
      const int nbytes = static_cast<int>(
          spatial_volume * static_cast<int64_t>(sizeof(MatrixT)));
      MPI_Comm tc = tComm();
      auto recv_poly = this->recv_poly;
      for (int stride = 1; stride < t_nproc; stride *= 2) {
        const int rel = t_coord % (2 * stride);
        if (rel == 0) {
          const int partner = t_coord + stride;
          if (partner >= t_nproc) {
            continue; // no partner at this level (non power of two)
          }
          void *rbuf = dev_mpi ? static_cast<void *>(recv_poly.data())
                               : static_cast<void *>(host_recv.data());
          MPI_Recv(rbuf, nbytes, MPI_BYTE, partner, 8100 + stride, tc,
                   MPI_STATUS_IGNORE);
          if (!dev_mpi) {
            Kokkos::deep_copy(recv_poly, host_recv);
          }
          Kokkos::parallel_for(
              "PolyakovLoop_combine", RangePolicy(0, spatial_volume),
              KOKKOS_LAMBDA(const int64_t s) {
                local_poly(s) *= recv_poly(s);
              });
          Kokkos::fence();
        } else if (rel == stride) {
          const int partner = t_coord - stride;
          const void *sbuf = static_cast<const void *>(local_poly.data());
          if (!dev_mpi) {
            Kokkos::deep_copy(host_send, local_poly);
            sbuf = static_cast<const void *>(host_send.data());
          }
          MPI_Send(sbuf, nbytes, MPI_BYTE, partner, 8100 + stride, tc);
          holds_result = false;
          break;
        }
      }
#endif

      if (holds_result) {
        // Single-value reduces only: HIP's CombinedReducer template for
        // (re, im) is disproportionately expensive to compile at large Nc.
        Kokkos::parallel_reduce(
            "PolyakovLoop_trace_re", RangePolicy(0, spatial_volume),
            KOKKOS_LAMBDA(const int64_t s, Real &reSum) {
              reSum += local_poly(s).trace().real();
            },
            poly_re);
        Kokkos::parallel_reduce(
            "PolyakovLoop_trace_im", RangePolicy(0, spatial_volume),
            KOKKOS_LAMBDA(const int64_t s, Real &imSum) {
              imSum += local_poly(s).trace().imag();
            },
            poly_im);
        poly_re /= Real(NCOLORS);
        poly_im /= Real(NCOLORS);
      }
    } else {
      // Product in a parallel_for (cheaper for HIP to compile than a
      // CombinedReducer parallel_reduce that also owns the matrices), then
      // two scalar reductions of the per-site traces.
      using TraceView = Kokkos::View<ComplexT *, DefaultMemSpace>;
      TraceView traces(Kokkos::view_alloc("PolyakovLoop_traces",
                                          Kokkos::WithoutInitializing),
                       spatial_volume);
      Kokkos::parallel_for(
          "PolyakovLoop_product", RangePolicy(0, spatial_volume),
          KOKKOS_LAMBDA(const int64_t spatialIdx) {
            ComplexT *gauge_ptr = gauge_view.data();

            int x[NDIMS];
            int64_t temp = spatialIdx;
            for (int i = 0; i < NDIMS - 1; ++i) {
              x[i] = static_cast<int>(temp % params.grid[i]);
              temp /= params.grid[i];
            }
            x[t_dir] = 0;

            MatrixT poly = MatrixT::identity();
            for (int t = 0; t < nt; ++t) {
              x[t_dir] = t;
              const int64_t idx_eo = coordsToEoIdx(x, params);
              MatrixT uT;
              loadGaugeLinkSoa(gauge_ptr, idx_eo, t_dir, size, params, uT, atype);
              poly *= uT;
            }
            traces(spatialIdx) = poly.trace() / Real(NCOLORS);
          });
      Kokkos::parallel_reduce(
          "PolyakovLoop_re", RangePolicy(0, spatial_volume),
          KOKKOS_LAMBDA(const int64_t s, Real &reSum) {
            reSum += traces(s).real();
          },
          poly_re);
      Kokkos::parallel_reduce(
          "PolyakovLoop_im", RangePolicy(0, spatial_volume),
          KOKKOS_LAMBDA(const int64_t s, Real &imSum) {
            imSum += traces(s).imag();
          },
          poly_im);
    }

#ifdef KWQFT_USE_MPI
    if (params.mpi) {
      double lr[2] = {static_cast<double>(poly_re), static_cast<double>(poly_im)};
      double gr[2];
      MPI_Allreduce(lr, gr, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      int64_t global_spatial = 1;
      for (int i = 0; i < NDIMS - 1; ++i) {
        global_spatial *= static_cast<int64_t>(params.global_grid[i]);
      }
      poly_value = ComplexT(
          static_cast<Real>(gr[0] / static_cast<double>(global_spatial)),
          static_cast<Real>(gr[1] / static_cast<double>(global_spatial)));
    } else
#endif
    {
      poly_value = ComplexT(poly_re / spatial_volume, poly_im / spatial_volume);
    }

    m_time = timer.seconds();
  }

  ComplexT value() const { return poly_value; }
  Real absValue() const { return poly_value.abs(); }
  double time() const { return m_time; }

  /**
   * @brief Calculate number of floating point operations
   */
  long long flop() const {
    int nt = params.grid[NDIMS - 1];
    long long spatial_volume = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      spatial_volume *= params.grid[i];
    }
    // nt products with the temporal links, then the real and imaginary parts
    // of the trace.
    return (static_cast<long long>(nt) * sunProductFlops() +
            2LL * (NCOLORS - 1)) *
           spatial_volume;
  }

  /**
   * @brief Calculate bytes read
   */
  long long bytes() const {
    int nt = params.grid[NDIMS - 1];
    long long spatial_volume = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      spatial_volume *= params.grid[i];
    }
    int num_params = gaugeNumParams(gauge.type());
    return spatial_volume * (num_params * nt + 2LL) * sizeof(Real);
  }

  double flops() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    return report.gflops;
  }

  double bandwidth() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    return report.bandwidth_gbs;
  }

  void printValue() const {
    if (params.mpi && mpiCommRank() != 0) {
      return;
    }
    printf("Polyakov Loop: %.12f + %.12f i (|P| = %.12f)\n",
           static_cast<double>(poly_value.real()),
           static_cast<double>(poly_value.imag()),
           static_cast<double>(absValue()));
  }

  void stat() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    if (mpiCommRank() != 0) {
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
  GaugeT &gauge;
  LatticeParams params;
  double m_time;

public:
  /**
   * @brief Gram-Schmidt reunitarization for a single matrix
   */
  KOKKOS_INLINE_FUNCTION
  static void reunitarizeMatrix(MatrixT &u) {
    if constexpr (NCOLORS == 3) {
      // For SU(3), use the simplified method
      // Normalize first row
      Real norm = Real(0);
      for (int j = 0; j < 3; ++j) {
        norm += u.e[0][j].abs2();
      }
      norm = Real(1) / Kokkos::sqrt(norm);
      for (int j = 0; j < 3; ++j) {
        u.e[0][j] *= norm;
      }

      // Orthogonalize second row to first
      ComplexT dot = ComplexT::zero();
      for (int j = 0; j < 3; ++j) {
        dot += ~u.e[0][j] * u.e[1][j];
      }
      for (int j = 0; j < 3; ++j) {
        u.e[1][j] -= dot * u.e[0][j];
      }

      // Normalize second row
      norm = Real(0);
      for (int j = 0; j < 3; ++j) {
        norm += u.e[1][j].abs2();
      }
      norm = Real(1) / Kokkos::sqrt(norm);
      for (int j = 0; j < 3; ++j) {
        u.e[1][j] *= norm;
      }

      // Third row is cross product of first two
      u.e[2][0] = ~(u.e[0][1] * u.e[1][2] - u.e[0][2] * u.e[1][1]);
      u.e[2][1] = ~(u.e[0][2] * u.e[1][0] - u.e[0][0] * u.e[1][2]);
      u.e[2][2] = ~(u.e[0][0] * u.e[1][1] - u.e[0][1] * u.e[1][0]);
    } else {
      // General Gram-Schmidt for SU(N)
      for (int row = 0; row < NCOLORS; ++row) {
        // Orthogonalize against previous rows
        for (int prev = 0; prev < row; ++prev) {
          ComplexT dot = ComplexT::zero();
          for (int j = 0; j < NCOLORS; ++j) {
            dot += ~u.e[prev][j] * u.e[row][j];
          }
          for (int j = 0; j < NCOLORS; ++j) {
            u.e[row][j] -= dot * u.e[prev][j];
          }
        }

        // Normalize
        Real norm = Real(0);
        for (int j = 0; j < NCOLORS; ++j) {
          norm += u.e[row][j].abs2();
        }
        norm = Real(1) / Kokkos::sqrt(norm);
        for (int j = 0; j < NCOLORS; ++j) {
          u.e[row][j] *= norm;
        }
      }
    }
  }

  /// Functor form of \ref reunitarizeMatrix (for ghost buffers).
  struct MatrixFunctor {
    KOKKOS_INLINE_FUNCTION void operator()(MatrixT &u) const {
      reunitarizeMatrix(u);
    }
  };

  Reunitarize(GaugeT &gauge, const LatticeParams &params)
      : gauge(gauge), params(params), m_time(0) {}

  /**
   * @brief Reunitarize all links
   *
   * MPI: the same deterministic link-wise map is applied to the ghost copies,
   * so the shared halo stays valid and no re-exchange is needed afterwards.
   */
  void run() {
    Kokkos::Timer timer;

    auto gauge_view = gauge.getView();
    int64_t size = gauge.size();      // volume * NDIMS
    int64_t total_links = params.size; // volume * NDIMS
    const ArrayType atype = gauge.type();

    Kokkos::parallel_for(
        "Reunitarize", RangePolicy(0, total_links),
        KOKKOS_LAMBDA(const int64_t linkIdx) {
          ComplexT *gauge_ptr = gauge_view.data();

          MatrixT U;
          loadGaugeMatrix(gauge_ptr, linkIdx, size, atype, U);
          reunitarizeMatrix(U);
          storeGaugeMatrix(gauge_ptr, linkIdx, size, atype, U);
        });
    Kokkos::fence();

    if (GaugeHaloBuffers<Real> *halo = gauge.halo(params)) {
      if (halo->allValid()) {
        halo->applyToGhosts(MatrixFunctor{});
      }
    }

    m_time = timer.seconds();
  }

  double time() const { return m_time; }

  /**
   * @brief Calculate number of floating point operations
   */
  long long flop() const {
    const long long n = NCOLORS;
    long long flop_per_link = 0;
    if (n == 3) {
      // Row norms, one conjugated dot product, one axpy, and the cross
      // product that rebuilds the third row.
      flop_per_link = 132;
    } else {
      // Row r is orthogonalized against r previous rows, then normalized.
      // A conjugated dot is N complex multiplies, N conjugations and (N-1)
      // complex additions; the axpy is N complex multiplies and N additions.
      for (long long row = 0; row < n; ++row) {
        flop_per_link += row * ((9 * n - 2) + 8 * n);
        flop_per_link += 6 * n + 1;
      }
    }
    return flop_per_link * params.size;
  }

  /**
   * @brief Calculate bytes read/written
   */
  long long bytes() const {
    int num_params = gaugeNumParams(gauge.type());
    // Read + write one matrix per link
    return 2LL * num_params * sizeof(Real) * params.size;
  }

  double flops() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    return report.gflops;
  }

  double bandwidth() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    return report.bandwidth_gbs;
  }

  void stat() const {
    const auto report =
        makePerfReport(flop(), bytes(), m_time, params.mpi, params.nproc);
    if (mpiCommRank() != 0) {
      return;
    }
    printf("Reunitarize:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", report.time,
           report.bandwidth_gbs, report.gflops);
  }
};

// Explicit instantiations live in src/{plaquette,polyakov,reunitarize}.cpp.
// Without these declarations every TU that calls run() (e.g. heatbath_main)
// recompiles the device kernels — the dominant HIP compile cost at large Nc.
extern template class Plaquette<double>;
extern template class PolyakovLoop<double>;
extern template class Reunitarize<double>;
#ifndef KOKKOS_ENABLE_HIP
extern template class Plaquette<float>;
extern template class PolyakovLoop<float>;
extern template class Reunitarize<float>;
#endif

} // namespace kwqft

#endif // KWQFT_MEASUREMENTS_HPP
