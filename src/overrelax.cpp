/**
 * @file overrelax.cpp
 * @brief Overrelaxation instantiation, split from monte.cpp for HIP compile memory.
 */

#include "monte.hpp"

namespace kwqft {

template class Overrelaxation<double>;
#ifndef KOKKOS_ENABLE_HIP
template class Overrelaxation<float>;
#endif

} // namespace kwqft
