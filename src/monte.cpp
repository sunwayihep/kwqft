/**
 * @file monte.cpp
 * @brief Implementation of Monte Carlo algorithms
 */

#include "monte.hpp"

namespace kwqft {

template class HeatBath<double>;
template class Overrelaxation<double>;
#ifndef KOKKOS_ENABLE_HIP
// HIP device compile of SU(N) kernels dominates wall time at large Nc; float
// is unused by heatbath/tests and is omitted there to cut compile work in half.
template class HeatBath<float>;
template class Overrelaxation<float>;
#endif

} // namespace kwqft
