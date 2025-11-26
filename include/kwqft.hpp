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

#include "kwqft_common.hpp"
#include "complex.hpp"
#include "msu2.hpp"
#include "matrixsun.hpp"
#include "constants.hpp"
#include "index.hpp"
#include "gauge_array.hpp"
#include "random.hpp"
#include "monte.hpp"
#include "measurements.hpp"

namespace kwqft {

/**
 * @brief Initialize KWQFT library
 * 
 * This should be called after Kokkos::initialize() and before
 * any KWQFT operations
 */
inline void initialize(int argc = 0, char* argv[] = nullptr) {
    // Print library info
    printf("==========================================================\n");
    printf("KWQFT - Kokkos Ken Wilson Quantum Field Theory Library\n");
    printf("SU(%d) gauge theory in %d dimensions\n", NCOLORS, NDIMS);
    printf("Execution space: %s\n", 
           typeid(DefaultExecSpace).name());
    printf("Memory space: %s\n",
           typeid(DefaultMemSpace).name());
    printf("==========================================================\n");
}

/**
 * @brief Finalize KWQFT library
 * 
 * Call before Kokkos::finalize()
 */
inline void finalize() {
    // Release Kokkos views before Kokkos::finalize()
    finalizeParams();
    printf("==========================================================\n");
    printf("KWQFT finalized\n");
    printf("==========================================================\n");
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

