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
 * with MPI) `mpiEnvInit()`, so main() can stay compact.
 */
inline void initialize(int argc = 0, char *argv[] = nullptr) {
#ifdef KWQFT_USE_MPI
  mpiEnvInit(&argc, &argv);
#endif

  Kokkos::initialize(argc, argv);

  // Print library info
  if (mpiCommRank() == 0) {
    printf("==========================================================\n");
    printf("KWQFT - Kokkos Ken Wilson Quantum Field Theory Library\n");
    printf("SU(%d) gauge theory in %d dimensions\n", NCOLORS, NDIMS);
    printf("Execution space: %s\n", typeid(DefaultExecSpace).name());
    printf("Memory space: %s\n", typeid(DefaultMemSpace).name());
#ifdef KWQFT_USE_MPI
#ifdef KWQFT_MPI_DEVICE_AWARE
    printf("MPI halo: DefaultMemSpace buffers (KWQFT_MPI_DEVICE_AWARE)\n");
#else
    printf("MPI halo: DefaultMemSpace if host-accessible, else pinned Host "
           "staging\n");
#endif
#endif
    printf("==========================================================\n");
  }
}

/**
 * @brief Finalize KWQFT library
 *
 * This function calls `finalizeParams()`, then (when built with MPI)
 * `mpiEnvFinalize()`, and finally `Kokkos::finalize()`.
 */
inline void finalize() {
  // Release Kokkos views before Kokkos::finalize()
  finalizeParams();
  if (mpiCommRank() == 0) {
    printf("==========================================================\n");
    printf("KWQFT finalized\n");
    printf("==========================================================\n");
  }

#ifdef KWQFT_USE_MPI
  mpiEnvFinalize();
#endif

  Kokkos::finalize();
}

/**
 * @brief Timer class for performance measurements
 */
class Timer {
private:
  Kokkos::Timer time_r;
  double m_elapsed;
  bool running;

public:
  Timer() : m_elapsed(0), running(false) {}

  void start() {
    time_r.reset();
    running = true;
  }

  void stop() {
    if (running) {
      m_elapsed = time_r.seconds();
      running = false;
    }
  }

  void reset() {
    m_elapsed = 0;
    running = false;
  }

  double elapsed() const {
    if (running) {
      return time_r.seconds();
    }
    return m_elapsed;
  }

  double getElapsedTime() const { return elapsed(); }
};

} // namespace kwqft

#endif // KWQFT_HPP
