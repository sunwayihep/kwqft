/**
 * @file reunitarize.cpp
 * @brief Implementation of reunitarization
 */

#include "measurements.hpp"

namespace kwqft {

template class Reunitarize<double>;
#ifndef KOKKOS_ENABLE_HIP
template class Reunitarize<float>;
#endif

} // namespace kwqft
