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
 * Writes full SU(N) matrices per link (Nc×Nc complex) in normal site order.
 * If the in-memory gauge uses SOA12, the third row is reconstructed before
 * writing. Serial: local lattice; MPI: collective MPI-IO — every rank writes
 * its own sub-block through a subarray file view.
 *
 * @tparam Real         In-memory gauge precision
 * @tparam RealSaveConf On-disk precision
 */
template <typename Real, typename RealSaveConf>
void saveGaugeBinary(const GaugeArray<Real> &gauge,
                       const std::string &filename);

/**
 * @brief Load gauge configuration from CULQCD/sunw binary layout.
 *
 * Inverse of \ref saveGaugeBinary. MPI: collective MPI-IO — every rank reads
 * its own sub-block directly (no rank-0 distribution).
 */
template <typename Real, typename RealSaveConf>
void loadGaugeBinary(GaugeArray<Real> &gauge, const std::string &filename);

} // namespace kwqft

#endif // KWQFT_IO_GAUGE_HPP
