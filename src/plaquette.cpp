/**
 * @file plaquette.cpp
 * @brief Implementation of plaquette measurement
 */

#include "measurements.hpp"

namespace kwqft {

template class Plaquette<double>;
#ifndef KOKKOS_ENABLE_HIP
template class Plaquette<float>;
#endif

} // namespace kwqft
