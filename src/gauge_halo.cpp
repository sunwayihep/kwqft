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
  face_vol_.resize(static_cast<size_t>(NDIMS));
  for (int mu = 0; mu < NDIMS; ++mu) {
    const int64_t fv = p.volume / static_cast<int64_t>(p.grid[mu]);
    face_vol_[mu] = fv;
    const size_t n = static_cast<size_t>(fv * NDIMS * mat_elems_);
    d_recv_m_[mu] = Kokkos::View<Complex<Real> *, Kokkos::LayoutRight,
                                  DefaultMemSpace>(Kokkos::view_alloc(
                                                       "d_recv_m", Kokkos::WithoutInitializing),
                                                   n);
    d_recv_p_[mu] = Kokkos::View<Complex<Real> *, Kokkos::LayoutRight,
                                  DefaultMemSpace>(Kokkos::view_alloc(
                                                       "d_recv_p", Kokkos::WithoutInitializing),
                                                   n);
    h_send_[mu] = Kokkos::View<Complex<Real> *, Kokkos::HostSpace>(
        Kokkos::view_alloc("h_send", Kokkos::WithoutInitializing), n);
    h_recv_[mu] = Kokkos::View<Complex<Real> *, Kokkos::HostSpace>(
        Kokkos::view_alloc("h_recv", Kokkos::WithoutInitializing), n);
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

  for (int mu = 0; mu < NDIMS; ++mu) {
    const int64_t fv = face_vol_[mu];
    const size_t n = static_cast<size_t>(fv * NDIMS * me);

    Kokkos::View<Complex<Real> *, Kokkos::LayoutRight, DefaultMemSpace> d_send(
        Kokkos::view_alloc("d_send_hi", Kokkos::WithoutInitializing), n);

    Kokkos::parallel_for(
        "pack_high_face",
        Kokkos::RangePolicy<DefaultExecSpace>(0, fv),
        KOKKOS_LAMBDA(const int64_t fidx) {
          int x[NDIMS];
          fidx_to_coords_high_face(fidx, mu, x, par);
          const int64_t idx_eo = coords_to_eo_idx(x, par);
          for (int dir = 0; dir < NDIMS; ++dir) {
            const int64_t off =
                (fidx * NDIMS + static_cast<int64_t>(dir)) * me;
            for (int i = 0; i < NCOLORS; ++i) {
              for (int j = 0; j < NCOLORS; ++j) {
                const int ij = j + i * NCOLORS;
                d_send(off + ij) =
                    gauge_soa[idx_eo + static_cast<int64_t>(dir) * par.volume +
                              static_cast<int64_t>(ij) * soa_stride];
              }
            }
          }
        });
    Kokkos::fence();

    Kokkos::deep_copy(h_send_[mu], d_send);

    int src1 = 0, dst1 = 0;
    MPI_Cart_shift(comm, mu, 1, &src1, &dst1);

    const int tag0 = 400 + mu;
    const size_t nbytes = n * sizeof(Complex<Real>);
    MPI_Sendrecv(h_send_[mu].data(), static_cast<int>(nbytes), MPI_BYTE, dst1,
                 tag0, h_recv_[mu].data(), static_cast<int>(nbytes), MPI_BYTE,
                 src1, tag0, comm, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(d_recv_m_[mu], h_recv_[mu]);

    Kokkos::View<Complex<Real> *, Kokkos::LayoutRight, DefaultMemSpace> d_slow(
        Kokkos::view_alloc("d_send_lo", Kokkos::WithoutInitializing), n);

    Kokkos::parallel_for(
        "pack_low_face",
        Kokkos::RangePolicy<DefaultExecSpace>(0, fv),
        KOKKOS_LAMBDA(const int64_t fidx) {
          int x[NDIMS];
          fidx_to_coords_low_face(fidx, mu, x, par);
          const int64_t idx_eo = coords_to_eo_idx(x, par);
          for (int dir = 0; dir < NDIMS; ++dir) {
            const int64_t off =
                (fidx * NDIMS + static_cast<int64_t>(dir)) * me;
            for (int i = 0; i < NCOLORS; ++i) {
              for (int j = 0; j < NCOLORS; ++j) {
                const int ij = j + i * NCOLORS;
                d_slow(off + ij) =
                    gauge_soa[idx_eo + static_cast<int64_t>(dir) * par.volume +
                              static_cast<int64_t>(ij) * soa_stride];
              }
            }
          }
        });
    Kokkos::fence();

    Kokkos::deep_copy(h_send_[mu], d_slow);

    int src2 = 0, dst2 = 0;
    MPI_Cart_shift(comm, mu, -1, &src2, &dst2);

    const int tag1 = 500 + mu;
    MPI_Sendrecv(h_send_[mu].data(), static_cast<int>(nbytes), MPI_BYTE, dst2,
                 tag1, h_recv_[mu].data(), static_cast<int>(nbytes), MPI_BYTE,
                 src2, tag1, comm, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(d_recv_p_[mu], h_recv_[mu]);
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
  for (int mu = 0; mu < NDIMS; ++mu) {
    h.recv_m[mu] = d_recv_m_[mu].data();
    h.recv_p[mu] = d_recv_p_[mu].data();
  }
  return h;
}

template class GaugeHaloBuffers<double>;
template class GaugeHaloBuffers<float>;

} // namespace kwqft
