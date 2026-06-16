/**
 * @file io_gauge.hpp
 * @brief Gauge configuration I/O
 */

#ifndef KWQFT_IO_GAUGE_HPP
#define KWQFT_IO_GAUGE_HPP

#include "gauge_array.hpp"
#include <string>

namespace kwqft {

/**
 * @brief Save gauge configuration in binary layout.
 *
 * Serial: writes local lattice in normal site order (even/odd storage).
 * MPI: all ranks participate; rank 0 assembles the global lattice and writes.
 *
 * @tparam Real         In-memory gauge precision
 * @tparam RealSaveConf On-disk precision
 * @param withheader    If true, prepend grid, beta, and precision metadata
 */
template <typename Real, typename RealSaveConf>
void save_gauge_binary(const GaugeArray<Real> &gauge, const std::string &filename,
                       bool withheader = false);

/**
 * @brief Load gauge configuration from CULQCD/sunw binary layout.
 *
 * Inverse of \ref save_gauge_binary. MPI: rank 0 reads and distributes to ranks.
 */
template <typename Real, typename RealSaveConf>
void load_gauge_binary(GaugeArray<Real> &gauge, const std::string &filename,
                       bool withheader = false);

} // namespace kwqft

#endif // KWQFT_IO_GAUGE_HPP
