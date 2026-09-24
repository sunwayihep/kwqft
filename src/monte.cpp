/**
 * @file monte.cpp
 * @brief Heatbath instantiation. Overrelaxation is a separate TU so a HIP
 * device compile of large Nc does not hold both kernels in one process.
 */

#include "monte.hpp"

namespace kwqft {

template class HeatBath<double>;
#ifndef KOKKOS_ENABLE_HIP
template class HeatBath<float>;
#endif

} // namespace kwqft
