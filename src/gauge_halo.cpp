/**
 * @file gauge_halo.cpp
 * @brief MPI halo exchange for gauge SOA (Kokkos pack + portable MPI path)
 */

#include "gauge_halo.hpp"
#include "index.hpp"
#include "neighbor_access.hpp"

#ifdef KWQFT_USE_MPI
#include "mpi_layout.hpp"
#endif

#include <algorithm>
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

/// Pack work item: one (dir, parity-or-both) block of one region.
/// Lives in the enclosing namespace so the CUDA lambda can copy it by value.
struct HaloPackItem {
  int dir;
  int64_t slot0;
  int64_t nslots;
};

constexpr int HALO_MAX_PACK_ITEMS = 2 * NDIMS;

struct HaloOffArr {
  int v[NDIMS];
};

template <typename Real>
GaugeHaloBuffers<Real>::GaugeHaloBuffers(const LatticeParams &p) : p_(p) {
  mat_elems_ = static_cast<int64_t>(NCOLORS * NCOLORS);
  halo_vol_.assign(HALO_CODE_COUNT, 0);
  split_dim_.assign(HALO_CODE_COUNT, -1);
  active_.assign(HALO_CODE_COUNT, 0);
  d_recv_.resize(HALO_CODE_COUNT);
  d_send_.resize(HALO_CODE_COUNT);
  h_send_.resize(HALO_CODE_COUNT);
  h_recv_.resize(HALO_CODE_COUNT);
  valid_.assign(2 * NDIMS, 0);

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
    split_dim_[code] = halo_region_split_dim(off, p);
    const size_t n = static_cast<size_t>(hv * NDIMS * mat_elems_);
    d_recv_[code] = DeviceView(
        Kokkos::view_alloc("d_recv_halo", Kokkos::WithoutInitializing), n);
    d_send_[code] = DeviceView(
        Kokkos::view_alloc("d_send_halo", Kokkos::WithoutInitializing), n);
    if (need_host_stage) {
      h_send_[code] = StageView(
          Kokkos::view_alloc("h_send_halo", Kokkos::WithoutInitializing), n);
      h_recv_[code] = StageView(
          Kokkos::view_alloc("h_recv_halo", Kokkos::WithoutInitializing), n);
    }
  }
}

template <typename Real> GaugeHaloBuffers<Real>::~GaugeHaloBuffers() {
  // Never leave MPI requests dangling.
  end_exchange();
}

//=============================================================================
// Validity tracking
//=============================================================================

template <typename Real> void GaugeHaloBuffers<Real>::invalidate() {
  std::fill(valid_.begin(), valid_.end(), 0);
}

template <typename Real>
void GaugeHaloBuffers<Real>::mark_dirty(int dir, int parity) {
  if (dir < 0 || dir >= NDIMS) {
    return;
  }
  if (parity < 0) {
    valid_[2 * dir] = 0;
    valid_[2 * dir + 1] = 0;
  } else {
    valid_[2 * dir + (parity & 1)] = 0;
  }
}

template <typename Real> bool GaugeHaloBuffers<Real>::all_valid() const {
  for (char v : valid_) {
    if (!v) {
      return false;
    }
  }
  return true;
}

//=============================================================================
// Site lists (boundary / interior) for overlap
//=============================================================================

template <typename Real> void GaugeHaloBuffers<Real>::build_site_lists() {
  if (site_lists_built_) {
    return;
  }
  const LatticeParams &p = p_;
  for (int parity = 0; parity < 2; ++parity) {
    std::vector<int64_t> bnd, inr;
    bnd.reserve(static_cast<size_t>(p.half_volume / 4 + 1));
    inr.reserve(static_cast<size_t>(p.half_volume));
    for (int64_t id = 0; id < p.half_volume; ++id) {
      int x[NDIMS];
      indexNdEo(x, id, parity, p);
      bool on_face = false;
      for (int d = 0; d < NDIMS; ++d) {
        if (p.proc_grid[d] > 1 && (x[d] == 0 || x[d] == p.grid[d] - 1)) {
          on_face = true;
          break;
        }
      }
      (on_face ? bnd : inr).push_back(id);
    }
    boundary_[parity] = SiteList(
        Kokkos::view_alloc("halo_boundary_sites", Kokkos::WithoutInitializing),
        bnd.size());
    interior_[parity] = SiteList(
        Kokkos::view_alloc("halo_interior_sites", Kokkos::WithoutInitializing),
        inr.size());
    Kokkos::View<int64_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        hb(bnd.data(), bnd.size()), hi(inr.data(), inr.size());
    Kokkos::deep_copy(boundary_[parity], hb);
    Kokkos::deep_copy(interior_[parity], hi);
  }
  site_lists_built_ = true;
}

template <typename Real>
const typename GaugeHaloBuffers<Real>::SiteList &
GaugeHaloBuffers<Real>::boundary_sites(int parity) {
  build_site_lists();
  return boundary_[parity & 1];
}

template <typename Real>
const typename GaugeHaloBuffers<Real>::SiteList &
GaugeHaloBuffers<Real>::interior_sites(int parity) {
  build_site_lists();
  return interior_[parity & 1];
}

//=============================================================================
// Exchange
//=============================================================================

template <typename Real>
void GaugeHaloBuffers<Real>::collect_stale_chunks(
    std::vector<Chunk> &chunks) const {
  chunks.clear();
  const int64_t me = mat_elems_;
  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (!active_[code]) {
      continue;
    }
    const int64_t hv = halo_vol_[code];
    const bool split = split_dim_[code] >= 0;
    const int64_t block = split ? hv / 2 : hv;
    int nchunk = 0;
    bool open = false;
    int64_t cur_off = 0, cur_n = 0;
    for (int dir = 0; dir < NDIMS; ++dir) {
      const int nblk = split ? 2 : 1;
      for (int b = 0; b < nblk; ++b) {
        const bool stale =
            split ? !valid_[2 * dir + b]
                  : (!valid_[2 * dir] || !valid_[2 * dir + 1]);
        const int64_t off =
            (static_cast<int64_t>(dir) * hv + static_cast<int64_t>(b) * block) *
            me;
        const int64_t n = block * me;
        if (stale) {
          if (open && cur_off + cur_n == off) {
            cur_n += n;
          } else {
            if (open) {
              chunks.push_back(Chunk{code, cur_off, cur_n,
                                     2000 + code * 32 + nchunk++});
            }
            cur_off = off;
            cur_n = n;
            open = true;
          }
        }
      }
    }
    if (open) {
      chunks.push_back(Chunk{code, cur_off, cur_n, 2000 + code * 32 + nchunk++});
    }
  }
}

template <typename Real>
void GaugeHaloBuffers<Real>::pack_stale(const ComplexT *gauge_soa,
                                        int64_t soa_stride) {
  const int64_t me = mat_elems_;
  const LatticeParams par = p_;
  bool launched = false;

  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (!active_[code]) {
      continue;
    }
    int off[NDIMS];
    halo_code_to_offset(code, off);
    const int64_t hv = halo_vol_[code];
    const int sdim = split_dim_[code];
    const bool split = sdim >= 0;

    Kokkos::Array<HaloPackItem, HALO_MAX_PACK_ITEMS> items;
    int nitems = 0;
    for (int dir = 0; dir < NDIMS; ++dir) {
      if (split) {
        for (int b = 0; b < 2; ++b) {
          if (!valid_[2 * dir + b]) {
            items[nitems++] = HaloPackItem{
                dir, static_cast<int64_t>(b) * (hv / 2), hv / 2};
          }
        }
      } else if (!valid_[2 * dir] || !valid_[2 * dir + 1]) {
        items[nitems++] = HaloPackItem{dir, 0, hv};
      }
    }
    if (nitems == 0) {
      continue;
    }
    // All items of one region have the same size.
    const int64_t item_n = items[0].nslots;
    auto d_send_ptr = d_send_[code].data();
    HaloOffArr offa;
    for (int d = 0; d < NDIMS; ++d) {
      offa.v[d] = off[d];
    }

    Kokkos::parallel_for(
        "pack_halo_region",
        Kokkos::RangePolicy<DefaultExecSpace>(0, item_n * nitems),
        KOKKOS_LAMBDA(const int64_t tid) {
          const int k = static_cast<int>(tid / item_n);
          const int64_t sub = tid - static_cast<int64_t>(k) * item_n;
          const HaloPackItem pi = items[k];
          const int64_t slot = pi.slot0 + sub;
          int x[NDIMS];
          halo_slot_to_coords(offa.v, sdim, hv, slot, x, par);
          const int64_t idx_eo = coords_to_eo_idx(x, par);
          const int64_t base =
              (static_cast<int64_t>(pi.dir) * hv + slot) * me;
          const int64_t src =
              idx_eo + static_cast<int64_t>(pi.dir) * par.volume;
          for (int ij = 0; ij < NCOLORS * NCOLORS; ++ij) {
            d_send_ptr[base + ij] =
                gauge_soa[src + static_cast<int64_t>(ij) * soa_stride];
          }
        });
    launched = true;
  }
  if (launched) {
    Kokkos::fence();
  }
}

template <typename Real>
void GaugeHaloBuffers<Real>::post_chunks(const std::vector<Chunk> &chunks) {
#ifdef KWQFT_USE_MPI
  MPI_Comm comm = kwqft_mpi_cart_comm();
  if (comm == MPI_COMM_NULL) {
    return;
  }
  const LatticeParams &p = p_;
  int my_coords[NDIMS];
  MPI_Cart_coords(comm, p.rank, NDIMS, my_coords);
  constexpr bool use_default_mem = mpi_default_mem();

  flight_reqs_.clear();
  flight_reqs_.reserve(chunks.size() * 2);

  if constexpr (!use_default_mem) {
    for (const auto &c : chunks) {
      const auto rng = std::make_pair(c.elem_off, c.elem_off + c.nelem);
      Kokkos::deep_copy(Kokkos::subview(h_send_[c.code], rng),
                        Kokkos::subview(d_send_[c.code], rng));
    }
  }

  for (const auto &c : chunks) {
    int off[NDIMS];
    halo_code_to_offset(c.code, off);
    int src_coords[NDIMS], dst_coords[NDIMS];
    for (int d = 0; d < NDIMS; ++d) {
      const int pd = p.proc_grid[d];
      src_coords[d] = ((my_coords[d] + off[d]) % pd + pd) % pd;
      dst_coords[d] = ((my_coords[d] - off[d]) % pd + pd) % pd;
    }
    int src_rank = 0, dst_rank = 0;
    MPI_Cart_rank(comm, src_coords, &src_rank);
    MPI_Cart_rank(comm, dst_coords, &dst_rank);

    const int64_t byte_off = c.elem_off * static_cast<int64_t>(sizeof(ComplexT));
    const int nbytes = static_cast<int>(c.nelem * sizeof(ComplexT));

    char *send_base;
    char *recv_base;
    if constexpr (use_default_mem) {
      send_base = reinterpret_cast<char *>(d_send_[c.code].data());
      recv_base = reinterpret_cast<char *>(d_recv_[c.code].data());
    } else {
      send_base = reinterpret_cast<char *>(h_send_[c.code].data());
      recv_base = reinterpret_cast<char *>(h_recv_[c.code].data());
    }

    if (src_rank == p.rank && dst_rank == p.rank) {
      // Self-neighbor (cannot happen for active codes, kept for safety).
      if constexpr (use_default_mem) {
        const auto rng = std::make_pair(c.elem_off, c.elem_off + c.nelem);
        Kokkos::deep_copy(Kokkos::subview(d_recv_[c.code], rng),
                          Kokkos::subview(d_send_[c.code], rng));
      } else {
        std::memcpy(recv_base + byte_off, send_base + byte_off,
                    static_cast<size_t>(nbytes));
      }
      continue;
    }

    MPI_Request rq;
    MPI_Irecv(recv_base + byte_off, nbytes, MPI_BYTE, src_rank, c.tag, comm,
              &rq);
    flight_reqs_.push_back(rq);
    MPI_Isend(send_base + byte_off, nbytes, MPI_BYTE, dst_rank, c.tag, comm,
              &rq);
    flight_reqs_.push_back(rq);
  }
#else
  (void)chunks;
#endif
}

template <typename Real>
void GaugeHaloBuffers<Real>::begin_exchange(const ComplexT *gauge_soa,
                                            int64_t soa_stride, int dir,
                                            int parity) {
  end_exchange();
  if (dir >= 0) {
    mark_dirty(dir, parity);
  }
  if (!p_.mpi || p_.nproc <= 1) {
    std::fill(valid_.begin(), valid_.end(), 1);
    return;
  }
  collect_stale_chunks(flight_chunks_);
  if (flight_chunks_.empty()) {
    return;
  }
  pack_stale(gauge_soa, soa_stride);
  post_chunks(flight_chunks_);
  in_flight_ = true;
  // Blocks are valid once end_exchange() returns; mark now so that a
  // subsequent mark_dirty() on another block is not lost.
  std::fill(valid_.begin(), valid_.end(), 1);
}

template <typename Real> void GaugeHaloBuffers<Real>::end_exchange() {
  if (!in_flight_) {
    return;
  }
  in_flight_ = false;
#ifdef KWQFT_USE_MPI
  if (!flight_reqs_.empty()) {
    MPI_Waitall(static_cast<int>(flight_reqs_.size()), flight_reqs_.data(),
                MPI_STATUSES_IGNORE);
    flight_reqs_.clear();
  }
  if constexpr (!mpi_default_mem()) {
    for (const auto &c : flight_chunks_) {
      const auto rng = std::make_pair(c.elem_off, c.elem_off + c.nelem);
      Kokkos::deep_copy(Kokkos::subview(d_recv_[c.code], rng),
                        Kokkos::subview(h_recv_[c.code], rng));
    }
  }
#endif
  // Device-aware MPI completion is not on the Kokkos stream; make the
  // newly arrived ghosts visible to the next kernel. Also waits for any
  // interior kernel that was overlapping the Waitall.
  Kokkos::fence();
  flight_chunks_.clear();
}

template <typename Real>
void GaugeHaloBuffers<Real>::refresh(const ComplexT *gauge_soa,
                                     int64_t soa_stride) {
  begin_exchange(gauge_soa, soa_stride, -1, -1);
  end_exchange();
}

template <typename Real>
GaugeHaloDevice<Real> GaugeHaloBuffers<Real>::device_view() const {
  GaugeHaloDevice<Real> h;
  for (int code = 0; code < HALO_CODE_COUNT; ++code) {
    if (code == HALO_CENTER_CODE || !active_[code]) {
      h.recv[code] = nullptr;
      h.vol[code] = 0;
      h.split_dim[code] = -1;
    } else {
      h.recv[code] = d_recv_[code].data();
      h.vol[code] = halo_vol_[code];
      h.split_dim[code] = split_dim_[code];
    }
  }
  return h;
}

template class GaugeHaloBuffers<double>;
template class GaugeHaloBuffers<float>;

} // namespace kwqft
