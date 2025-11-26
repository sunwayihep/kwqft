/**
 * @file polyakov.cpp
 * @brief Implementation of Polyakov loop measurement
 */

#include "measurements.hpp"

namespace kwqft {

// Explicit template instantiations
template class PolyakovLoop<float>;
template class PolyakovLoop<double>;

} // namespace kwqft
