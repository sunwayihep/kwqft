/**
 * @file gauge_halo.hpp
 * @brief MPI halo buffers and exchange for SOA gauge (nearest-neighbor)
 */

#ifndef KWQFT_GAUGE_HALO_HPP
#define KWQFT_GAUGE_HALO_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "neighbor_access.hpp"
#include "kwqft_common.hpp"

namespace kwqft {

template <typename Real> class GaugeHaloBuffers {
public:
  using ComplexT = Complex<Real>;

  explicit GaugeHaloBuffers(const LatticeParams &p);

  /// Pack interior SOA gauge, exchange MPI, copy to device views (no-op if !p.mpi).
  void exchange(ComplexT *gauge_soa, int64_t soa_stride, const LatticeParams &p);

  GaugeHaloDevice<Real> device_view() const;

private:
  LatticeParams p_{};
  int64_t mat_elems_{0};
  std::vector<int64_t> face_vol_;
  Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace> d_recv_m_[NDIMS];
  Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace> d_recv_p_[NDIMS];
  Kokkos::View<ComplexT *, Kokkos::HostSpace> h_send_[NDIMS];
  Kokkos::View<ComplexT *, Kokkos::HostSpace> h_recv_[NDIMS];
};

} // namespace kwqft

#endif
