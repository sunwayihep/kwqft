/**
 * @file gauge_halo.cpp
 * @brief MPI halo exchange for gauge SOA (device pack, host MPI)
 */

#include "gauge_halo.hpp"
#include "neighbor_access.hpp"

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#include "mpi_layout.hpp"
#endif

#include <cstring>

namespace kwqft {

template <typename Real>
GaugeHaloBuffers<Real>::GaugeHaloBuffers(const LatticeParams &p) : p_(p) {
  mat_elems_ = static_cast<int64_t>(NCOLORS * NCOLORS);
  halo_vol_.assign(HALO_CODE_COUNT, 0);
  d_recv_.resize(HALO_CODE_COUNT);
  h_send_.resize(HALO_CODE_COUNT);
  h_recv_.resize(HALO_CODE_COUNT);

  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (code == HALO_CENTER_CODE) {
      continue;
    }
    int off[NDIMS];
    halo_code_to_offset(code, off);
    const int64_t hv = halo_region_volume(off, p);
    halo_vol_[code] = hv;
    const size_t n = static_cast<size_t>(hv * NDIMS * mat_elems_);
    d_recv_[code] = Kokkos::View<Complex<Real> *, Kokkos::LayoutRight,
                                 DefaultMemSpace>(
        Kokkos::view_alloc("d_recv_halo", Kokkos::WithoutInitializing), n);
    h_send_[code] = Kokkos::View<Complex<Real> *, Kokkos::HostSpace>(
        Kokkos::view_alloc("h_send_halo", Kokkos::WithoutInitializing), n);
    h_recv_[code] = Kokkos::View<Complex<Real> *, Kokkos::HostSpace>(
        Kokkos::view_alloc("h_recv_halo", Kokkos::WithoutInitializing), n);
  }
}

template <typename Real>
void GaugeHaloBuffers<Real>::exchange(Complex<Real> *gauge_soa, int64_t soa_stride,
                                      const LatticeParams &p) {
  if (!p.mpi || p.nproc <= 1) {
    return;
  }

#ifdef KWQFT_USE_MPI
  MPI_Comm comm = kwqft_mpi_cart_comm();
  if (comm == MPI_COMM_NULL) {
    return;
  }

  const int64_t me = mat_elems_;
  const LatticeParams par = p;
  int my_coords[NDIMS];
  MPI_Cart_coords(comm, p.rank, NDIMS, my_coords);

  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (code == HALO_CENTER_CODE) {
      continue;
    }
    int off[NDIMS];
    halo_code_to_offset(code, off);
    const int64_t hv = halo_vol_[code];
    const size_t n = static_cast<size_t>(hv * NDIMS * me);

    Kokkos::View<Complex<Real> *, Kokkos::LayoutRight, DefaultMemSpace> d_send(
        Kokkos::view_alloc("d_send_halo", Kokkos::WithoutInitializing), n);

    Kokkos::parallel_for(
        "pack_halo_region",
        Kokkos::RangePolicy<DefaultExecSpace>(0, hv),
        KOKKOS_LAMBDA(const int64_t ridx) {
          int x[NDIMS];
          int64_t t = ridx;
          for (int d = 0; d < NDIMS; ++d) {
            if (off[d] == -1) {
              x[d] = par.grid[d] - 1;
            } else if (off[d] == +1) {
              x[d] = 0;
            } else {
              x[d] = static_cast<int>(t % static_cast<int64_t>(par.grid[d]));
              t /= static_cast<int64_t>(par.grid[d]);
            }
          }
          const int64_t idx_eo = coords_to_eo_idx(x, par);
          for (int dir = 0; dir < NDIMS; ++dir) {
            const int64_t base =
                (ridx * NDIMS + static_cast<int64_t>(dir)) * me;
            for (int i = 0; i < NCOLORS; ++i) {
              for (int j = 0; j < NCOLORS; ++j) {
                const int ij = j + i * NCOLORS;
                d_send(base + ij) =
                    gauge_soa[idx_eo + static_cast<int64_t>(dir) * par.volume +
                              static_cast<int64_t>(ij) * soa_stride];
              }
            }
          }
        });
    Kokkos::fence();
    Kokkos::deep_copy(h_send_[code], d_send);

    int src_coords[NDIMS], dst_coords[NDIMS];
    for (int d = 0; d < NDIMS; ++d) {
      int pd = p.proc_grid[d];
      int s = my_coords[d] + off[d];
      int r = my_coords[d] - off[d];
      s = (s % pd + pd) % pd;
      r = (r % pd + pd) % pd;
      src_coords[d] = s;
      dst_coords[d] = r;
    }

    int src_rank = 0, dst_rank = 0;
    MPI_Cart_rank(comm, src_coords, &src_rank);
    MPI_Cart_rank(comm, dst_coords, &dst_rank);

    const int tag = 1000 + code;
    const int nbytes = static_cast<int>(n * sizeof(Complex<Real>));
    MPI_Sendrecv(h_send_[code].data(), nbytes, MPI_BYTE, dst_rank, tag,
                 h_recv_[code].data(), nbytes, MPI_BYTE, src_rank, tag, comm,
                 MPI_STATUS_IGNORE);

    Kokkos::deep_copy(d_recv_[code], h_recv_[code]);
  }
#else
  (void)gauge_soa;
  (void)soa_stride;
  (void)p;
#endif
}

template <typename Real>
GaugeHaloDevice<Real> GaugeHaloBuffers<Real>::device_view() const {
  GaugeHaloDevice<Real> h;
  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (code == HALO_CENTER_CODE) {
      h.recv[code] = nullptr;
    } else {
      h.recv[code] = d_recv_[code].data();
    }
  }
  return h;
}

template class GaugeHaloBuffers<double>;
template class GaugeHaloBuffers<float>;

} // namespace kwqft
