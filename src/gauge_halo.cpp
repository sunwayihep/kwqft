/**
 * @file gauge_halo.cpp
 * @brief MPI halo exchange for gauge SOA (Kokkos pack + portable MPI path)
 */

#include "gauge_halo.hpp"
#include "neighbor_access.hpp"

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#include "mpi_layout.hpp"
#include <vector>
#endif

#include <cstring>

namespace kwqft {

namespace {

bool halo_offset_needs_mpi(const int off[NDIMS], const LatticeParams &p) {
  int nnz = 0;
  int split_nnz = 0;
  for (int d = 0; d < NDIMS; ++d) {
    if (off[d] == 0) {
      continue;
    }
    nnz++;
    if (p.proc_grid[d] > 1) {
      split_nnz++;
    }
  }
  if (nnz == 0 || nnz > 2) {
    return false;
  }
  return split_nnz == nnz;
}

} // namespace

template <typename Real>
GaugeHaloBuffers<Real>::GaugeHaloBuffers(const LatticeParams &p) : p_(p) {
  mat_elems_ = static_cast<int64_t>(NCOLORS * NCOLORS);
  halo_vol_.assign(HALO_CODE_COUNT, 0);
  active_.assign(HALO_CODE_COUNT, 0);
  d_recv_.resize(HALO_CODE_COUNT);
  d_send_.resize(HALO_CODE_COUNT);
  h_send_.resize(HALO_CODE_COUNT);
  h_recv_.resize(HALO_CODE_COUNT);

  const bool need_host_stage = !mpi_default_mem();

  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (code == HALO_CENTER_CODE) {
      continue;
    }
    int off[NDIMS];
    halo_code_to_offset(code, off);
    if (!halo_offset_needs_mpi(off, p)) {
      continue;
    }
    active_[code] = 1;
    const int64_t hv = halo_region_volume(off, p);
    halo_vol_[code] = hv;
    const size_t n = static_cast<size_t>(hv * NDIMS * mat_elems_);
    d_recv_[code] = Kokkos::View<ComplexT *, Kokkos::LayoutRight,
                                 DefaultMemSpace>(
        Kokkos::view_alloc("d_recv_halo", Kokkos::WithoutInitializing), n);
    d_send_[code] = Kokkos::View<ComplexT *, Kokkos::LayoutRight,
                                 DefaultMemSpace>(
        Kokkos::view_alloc("d_send_halo", Kokkos::WithoutInitializing), n);
    if (need_host_stage) {
      h_send_[code] = Kokkos::View<ComplexT *, Kokkos::HostSpace>(
          Kokkos::view_alloc("h_send_halo", Kokkos::WithoutInitializing), n);
      h_recv_[code] = Kokkos::View<ComplexT *, Kokkos::HostSpace>(
          Kokkos::view_alloc("h_recv_halo", Kokkos::WithoutInitializing), n);
    }
  }
}

template <typename Real>
void GaugeHaloBuffers<Real>::exchange(ComplexT *gauge_soa, int64_t soa_stride,
                                      const LatticeParams &p) {
  exchange_dir_range(gauge_soa, soa_stride, p, 0, NDIMS);
}

template <typename Real>
void GaugeHaloBuffers<Real>::exchange_dir(ComplexT *gauge_soa,
                                          int64_t soa_stride,
                                          const LatticeParams &p, int mu) {
  if (mu < 0 || mu >= NDIMS) {
    return;
  }
  exchange_dir_range(gauge_soa, soa_stride, p, mu, mu + 1);
}

template <typename Real>
void GaugeHaloBuffers<Real>::exchange_dir_range(ComplexT *gauge_soa,
                                                int64_t soa_stride,
                                                const LatticeParams &p,
                                                int dir0, int dir1) {
  if (!p.mpi || p.nproc <= 1 || dir0 >= dir1) {
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
  constexpr bool use_default_mem = mpi_default_mem();

  struct Pending {
    int code;
    int src_rank;
    int dst_rank;
    int tag;
    int nbytes;
    int64_t byte_off;
  };
  std::vector<Pending> pending;
  pending.reserve(static_cast<size_t>(HALO_CODE_COUNT));

  // Pack all active regions (dir-major: dir * hv + ridx).
  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (!active_[code]) {
      continue;
    }
    int off[NDIMS];
    halo_code_to_offset(code, off);
    const int64_t hv = halo_vol_[code];
    auto d_send = d_send_[code];
    auto d_send_ptr = d_send.data();
    const int d0 = dir0;
    const int d1 = dir1;

    Kokkos::parallel_for(
        "pack_halo_region",
        Kokkos::RangePolicy<DefaultExecSpace>(0, hv * (d1 - d0)),
        KOKKOS_LAMBDA(const int64_t tid) {
          const int64_t ridx = tid % hv;
          const int dir = d0 + static_cast<int>(tid / hv);
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
          const int64_t base =
              (static_cast<int64_t>(dir) * hv + ridx) * me;
          for (int i = 0; i < NCOLORS; ++i) {
            for (int j = 0; j < NCOLORS; ++j) {
              const int ij = j + i * NCOLORS;
              d_send_ptr[base + ij] =
                  gauge_soa[idx_eo + static_cast<int64_t>(dir) * par.volume +
                            static_cast<int64_t>(ij) * soa_stride];
            }
          }
        });
  }
  Kokkos::fence();

  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (!active_[code]) {
      continue;
    }
    int off[NDIMS];
    halo_code_to_offset(code, off);
    const int64_t hv = halo_vol_[code];

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

    const int64_t elem_off = static_cast<int64_t>(dir0) * hv * me;
    const int64_t nelem =
        static_cast<int64_t>(dir1 - dir0) * hv * me;
    const int nbytes = static_cast<int>(nelem * sizeof(ComplexT));
    const int64_t byte_off = elem_off * static_cast<int64_t>(sizeof(ComplexT));
    const int tag = 2000 + code * (NDIMS + 1) + dir0;

    if (src_rank == p.rank && dst_rank == p.rank) {
      if constexpr (use_default_mem) {
        Kokkos::deep_copy(
            Kokkos::subview(d_recv_[code],
                            std::make_pair(elem_off, elem_off + nelem)),
            Kokkos::subview(d_send_[code],
                            std::make_pair(elem_off, elem_off + nelem)));
      } else {
        Kokkos::deep_copy(
            Kokkos::subview(h_send_[code],
                            std::make_pair(elem_off, elem_off + nelem)),
            Kokkos::subview(d_send_[code],
                            std::make_pair(elem_off, elem_off + nelem)));
        std::memcpy(
            reinterpret_cast<char *>(h_recv_[code].data()) + byte_off,
            reinterpret_cast<char *>(h_send_[code].data()) + byte_off,
            static_cast<size_t>(nbytes));
        Kokkos::deep_copy(
            Kokkos::subview(d_recv_[code],
                            std::make_pair(elem_off, elem_off + nelem)),
            Kokkos::subview(h_recv_[code],
                            std::make_pair(elem_off, elem_off + nelem)));
      }
      continue;
    }

    pending.push_back(
        Pending{code, src_rank, dst_rank, tag, nbytes, byte_off});
  }

  if (pending.empty()) {
    return;
  }

  if constexpr (!use_default_mem) {
    for (const auto &pr : pending) {
      const int64_t hv = halo_vol_[pr.code];
      const int64_t elem_off = static_cast<int64_t>(dir0) * hv * me;
      const int64_t nelem =
          static_cast<int64_t>(dir1 - dir0) * hv * me;
      Kokkos::deep_copy(
          Kokkos::subview(h_send_[pr.code],
                          std::make_pair(elem_off, elem_off + nelem)),
          Kokkos::subview(d_send_[pr.code],
                          std::make_pair(elem_off, elem_off + nelem)));
    }
  }

  std::vector<MPI_Request> reqs(pending.size() * 2);
  for (size_t i = 0; i < pending.size(); ++i) {
    const auto &pr = pending[i];
    void *send_ptr;
    void *recv_ptr;
    if constexpr (use_default_mem) {
      send_ptr = reinterpret_cast<char *>(d_send_[pr.code].data()) + pr.byte_off;
      recv_ptr = reinterpret_cast<char *>(d_recv_[pr.code].data()) + pr.byte_off;
    } else {
      send_ptr = reinterpret_cast<char *>(h_send_[pr.code].data()) + pr.byte_off;
      recv_ptr = reinterpret_cast<char *>(h_recv_[pr.code].data()) + pr.byte_off;
    }
    MPI_Irecv(recv_ptr, pr.nbytes, MPI_BYTE, pr.src_rank, pr.tag, comm,
              &reqs[2 * i]);
    MPI_Isend(send_ptr, pr.nbytes, MPI_BYTE, pr.dst_rank, pr.tag, comm,
              &reqs[2 * i + 1]);
  }
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  if constexpr (!use_default_mem) {
    for (const auto &pr : pending) {
      const int64_t hv = halo_vol_[pr.code];
      const int64_t elem_off = static_cast<int64_t>(dir0) * hv * me;
      const int64_t nelem =
          static_cast<int64_t>(dir1 - dir0) * hv * me;
      Kokkos::deep_copy(
          Kokkos::subview(d_recv_[pr.code],
                          std::make_pair(elem_off, elem_off + nelem)),
          Kokkos::subview(h_recv_[pr.code],
                          std::make_pair(elem_off, elem_off + nelem)));
    }
  }
#else
  (void)gauge_soa;
  (void)soa_stride;
  (void)p;
  (void)dir0;
  (void)dir1;
#endif
}

template <typename Real>
GaugeHaloDevice<Real> GaugeHaloBuffers<Real>::device_view() const {
  GaugeHaloDevice<Real> h;
  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (code == HALO_CENTER_CODE || !active_[code]) {
      h.recv[code] = nullptr;
      h.vol[code] = 0;
    } else {
      h.recv[code] = d_recv_[code].data();
      h.vol[code] = halo_vol_[code];
    }
  }
  return h;
}

template class GaugeHaloBuffers<double>;
template class GaugeHaloBuffers<float>;

} // namespace kwqft
