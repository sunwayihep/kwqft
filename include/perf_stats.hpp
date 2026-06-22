/**
 * @file perf_stats.hpp
 * @brief MPI-aware aggregation of kernel performance counters
 */

#ifndef KWQFT_PERF_STATS_HPP
#define KWQFT_PERF_STATS_HPP

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

} // namespace kwqft

#endif // KWQFT_PERF_STATS_HPP
