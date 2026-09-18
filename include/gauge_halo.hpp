/**
 * @file gauge_halo.hpp
 * @brief MPI face/edge halo buffers for SOA gauge (lazy shift / halo staple)
 *
 * Only regions that cross a domain-decomposed direction are allocated and
 * exchanged. Directions with \c proc_grid[d]==1 wrap locally and never use MPI.
 * Hypercube corners (3+ nonzero offsets) are omitted: Wilson staple/plaquette
 * need at most face+edge ghosts.
 *
 * Buffer layout is dir-major so a single link direction is one contiguous MPI
 * message. \ref exchange refreshes all directions; \ref exchange_dir only one
 * (used after each HB/OR mu update).
 *
 * With \c KWQFT_MPI_DEVICE_AWARE, MPI uses DefaultMemSpace pointers directly;
 * otherwise non-host-accessible memory is staged through HostSpace.
 */

#ifndef KWQFT_GAUGE_HALO_HPP
#define KWQFT_GAUGE_HALO_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "neighbor_access.hpp"
#include "kwqft_common.hpp"
#include <memory>
#include <vector>

namespace kwqft {

template <typename Real> class GaugeHaloBuffers {
public:
  using ComplexT = Complex<Real>;

  explicit GaugeHaloBuffers(const LatticeParams &p);

  /// Pack and exchange all link directions for active face/edge regions.
  void exchange(ComplexT *gauge_soa, int64_t soa_stride, const LatticeParams &p);

  /// Pack and exchange a single link direction (after that mu was updated).
  void exchange_dir(ComplexT *gauge_soa, int64_t soa_stride,
                    const LatticeParams &p, int mu);

  GaugeHaloDevice<Real> device_view() const;

  /// Pack/exchange link directions [dir0, dir1).
  void exchange_dir_range(ComplexT *gauge_soa, int64_t soa_stride,
                          const LatticeParams &p, int dir0, int dir1);

private:
  static constexpr bool mpi_default_mem() {
#if defined(KWQFT_MPI_DEVICE_AWARE)
    return true;
#else
    return Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                      DefaultMemSpace>::accessible;
#endif
  }

  LatticeParams p_{};
  int64_t mat_elems_{0};
  std::vector<int64_t> halo_vol_;
  std::vector<char> active_;
  std::vector<Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>>
      d_recv_;
  std::vector<Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>>
      d_send_;
  std::vector<Kokkos::View<ComplexT *, Kokkos::HostSpace>> h_send_;
  std::vector<Kokkos::View<ComplexT *, Kokkos::HostSpace>> h_recv_;
};

template <typename Real>
inline std::unique_ptr<GaugeHaloBuffers<Real>>
make_halo_if_mpi(const LatticeParams &params) {
  if (params.mpi && params.nproc > 1) {
    return std::make_unique<GaugeHaloBuffers<Real>>(params);
  }
  return nullptr;
}

} // namespace kwqft

#endif
