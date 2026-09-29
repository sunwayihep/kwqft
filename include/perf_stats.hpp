/**
 * @file perf_stats.hpp
 * @brief MPI-aware aggregation of kernel performance counters
 */

#ifndef KWQFT_PERF_STATS_HPP
#define KWQFT_PERF_STATS_HPP

#include "kwqft_common.hpp"

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif

namespace kwqft {

struct PerfReport {
  long long flop{0};
  long long bytes{0};
  double time{0.0};
  double gflops{0.0};
  double bandwidth_gbs{0.0};
};

/**
 * @brief Build a performance report from per-rank counters.
 *
 * Serial / single-rank: local values only.
 * MPI (nproc > 1): sum flop and bytes over ranks, use max(time) as wall clock.
 */
inline PerfReport make_perf_report(long long local_flop, long long local_bytes,
                                   double local_time, bool mpi_domain,
                                   int nproc) {
  long long flop = local_flop;
  long long bytes = local_bytes;
  double time = local_time;

#ifdef KWQFT_USE_MPI
  if (mpi_domain && nproc > 1) {
    const double flop_local = static_cast<double>(local_flop);
    const double bytes_local = static_cast<double>(local_bytes);
    double flop_sum = 0.0;
    double bytes_sum = 0.0;
    double time_max = 0.0;
    MPI_Allreduce(&flop_local, &flop_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&bytes_local, &bytes_sum, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_time, &time_max, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    flop = static_cast<long long>(flop_sum);
    bytes = static_cast<long long>(bytes_sum);
    time = time_max;
  }
#else
  (void)mpi_domain;
  (void)nproc;
#endif

  PerfReport report;
  report.flop = flop;
  report.bytes = bytes;
  report.time = time;
  if (time > 0.0) {
    report.gflops = static_cast<double>(flop) * 1.0e-9 / time;
    report.bandwidth_gbs =
        static_cast<double>(bytes) / (time * static_cast<double>(1LL << 30));
  }
  return report;
}

/**
 * @brief Flops in one SU(N) product.
 *
 * A complex multiplication is 6 real flops (4 multiplies and 2 adds) and a
 * complex addition is 2. The product has N^3 multiplications and N^2(N-1)
 * additions. A dagger only conjugates an operand and is not counted here.
 */
inline long long sun_product_flops() {
  const long long n = NCOLORS;
  return 6 * n * n * n + 2 * n * n * (n - 1);
}

/**
 * @brief Flops to build one Wilson staple.
 *
 * 2(D-1) legs, two SU(N) products per leg.
 */
inline long long staple_flops_per_link() {
  return 4LL * static_cast<long long>(NDIMS - 1) * sun_product_flops();
}

/**
 * @brief Flops in the Cabibbo--Marinari heatbath of one link, excluding the
 * staple.
 *
 * The scalar SU(2) draw is 46 flops: 4 adds to project the 2x2 block, the
 * quaternion norm (4 multiplies, 3 adds, a square root), one multiply by
 * beta, one reciprocal, 4 multiplies to rescale, and the quaternion product
 * U*V^dagger (16 multiplies and 12 adds). Embedding that SU(2) into two rows
 * is 4 complex multiplications and 2 complex additions per column.
 * SU(2) and SU(3) follow the specialized kernels, which do not form U*Sigma.
 */
inline long long heatbath_algebra_flops() {
  const long long n = NCOLORS;
  if (n == 2) {
    return 46;
  }
  if (n == 3) {
    // Four length-N dot products, then one 2-row embedding, per subgroup.
    return 3 * (60 * n + 38);
  }
  const long long nsub = n * (n - 1) / 2;
  return sun_product_flops() + nsub * (56 * n + 46);
}

/**
 * @brief Flops in the overrelaxation of one link, excluding the staple.
 *
 * SU(3) uses the specialized kernel (no full U*Sigma). Every other N forms
 * U*Sigma once and reflects each subgroup by two embeddings of U and of
 * U*Sigma.
 */
inline long long overrelax_algebra_flops() {
  const long long n = NCOLORS;
  if (n == 3) {
    return 3 * (88 * n + 12);
  }
  const long long nsub = n * (n - 1) / 2;
  return sun_product_flops() + nsub * (112 * n + 20);
}

/**
 * @brief Link matrices moved by one link update.
 *
 * The staple reads 6(D-1) neighboring links; the updated link is loaded
 * and stored. This is 20 at D=4.
 */
inline long long staple_links_per_update() {
  return 6LL * static_cast<long long>(NDIMS - 1) + 2LL;
}

} // namespace kwqft

#endif // KWQFT_PERF_STATS_HPP
