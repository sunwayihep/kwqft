/**
 * @file polyakov.cpp
 * @brief Implementation of Polyakov loop measurement
 */

#include "measurements.hpp"

namespace kwqft {

template class PolyakovLoop<double>;
#ifndef KOKKOS_ENABLE_HIP
template class PolyakovLoop<float>;
#endif

} // namespace kwqft
