/**
 * @file halo_exchange.hpp
 * @brief Halo exchange hook before batched shift (MPI extension point)
 *
 * Serial: no-op. With MPI, fill ghost cells of the local SOA gauge field so that
 * \ref shift_eo / dense shift reads at boundaries match the global lattice.
 */

#ifndef KWQFT_HALO_EXCHANGE_HPP
#define KWQFT_HALO_EXCHANGE_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "kwqft_common.hpp"

namespace kwqft {

template <typename Real>
inline void halo_exchange_gauge_soa_before_shift(Complex<Real> * /*gauge_soa*/,
                                                 int64_t /*soa_stride*/,
                                                 const LatticeParams & /*p*/) {
  // Serial: nothing. MPI: pack faces, MPI_Isend/Irecv, unpack into ghost zones.
}

} // namespace kwqft

#endif
