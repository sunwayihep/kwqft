/**
 * @file monte.cpp
 * @brief Implementation of Monte Carlo algorithms
 */

#include "monte.hpp"

namespace kwqft {

// Explicit template instantiations
template class HeatBath<float>;
template class HeatBath<double>;
template class Overrelaxation<float>;
template class Overrelaxation<double>;

} // namespace kwqft

