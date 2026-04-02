/**
 * @file kwqft.hpp
 * @brief Main header file for KWQFT - Kokkos Ken Wilson Quantum Field Theory
 *
 * KWQFT implements lattice gauge theory calculations using the Kokkos
 * programming model for performance portability across CPUs and GPUs.
 *
 * Include this single header to get all KWQFT functionality
 */

#ifndef KWQFT_HPP
#define KWQFT_HPP

#include <Kokkos_Core.hpp>

#ifdef KWQFT_USE_MPI
#include "mpi_layout.hpp"
#endif

#include "complex.hpp"
#include "constants.hpp"
#include "gauge_array.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"
#include "measurements.hpp"
#include "monte.hpp"
#include "msu2.hpp"
#include "random.hpp"

namespace kwqft {

/**
 * @brief Initialize KWQFT library
 *
 * This function also takes care of `Kokkos::initialize()` and (when built
 * with MPI) `mpi_env_init()`, so main() can stay compact.
 */
inline void initialize(int argc = 0, char *argv[] = nullptr) {
#ifdef KWQFT_USE_MPI
  mpi_env_init(&argc, &argv);
#endif

  Kokkos::initialize(argc, argv);

  // Print library info
  if (mpi_comm_rank() == 0) {
    printf("==========================================================\n");
    printf("KWQFT - Kokkos Ken Wilson Quantum Field Theory Library\n");
    printf("SU(%d) gauge theory in %d dimensions\n", NCOLORS, NDIMS);
    printf("Execution space: %s\n", typeid(DefaultExecSpace).name());
    printf("Memory space: %s\n", typeid(DefaultMemSpace).name());
    printf("==========================================================\n");
  }
}

/**
 * @brief Finalize KWQFT library
 *
 * This function calls `finalizeParams()`, then (when built with MPI)
 * `mpi_env_finalize()`, and finally `Kokkos::finalize()`.
 */
inline void finalize() {
  // Release Kokkos views before Kokkos::finalize()
  finalizeParams();
  if (mpi_comm_rank() == 0) {
    printf("==========================================================\n");
    printf("KWQFT finalized\n");
    printf("==========================================================\n");
  }

#ifdef KWQFT_USE_MPI
  mpi_env_finalize();
#endif

  Kokkos::finalize();
}

/**
 * @brief Timer class for performance measurements
 */
class Timer {
private:
  Kokkos::Timer time_r;
  double elapsed_;
  bool running_;

public:
  Timer() : elapsed_(0), running_(false) {}

  void start() {
    time_r.reset();
    running_ = true;
  }

  void stop() {
    if (running_) {
      elapsed_ = time_r.seconds();
      running_ = false;
    }
  }

  void reset() {
    elapsed_ = 0;
    running_ = false;
  }

  double elapsed() const {
    if (running_) {
      return time_r.seconds();
    }
    return elapsed_;
  }

  double get_elapsed_time() const { return elapsed(); }
};

} // namespace kwqft

#endif // KWQFT_HPP
