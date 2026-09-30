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

bool haloOffsetNeedsMpi(const int off[NDIMS], const LatticeParams &p) {
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

constexpr int halo_max_pack_items = 2 * NDIMS;

struct HaloOffArr {
  int v[NDIMS];
};

template <typename Real>
GaugeHaloBuffers<Real>::GaugeHaloBuffers(const LatticeParams &p) : p(p) {
  mat_elems = static_cast<int64_t>(NCOLORS * NCOLORS);
  halo_vol.assign(halo_code_count, 0);
  split_dim.assign(halo_code_count, -1);
  active.assign(halo_code_count, 0);
  d_recv.resize(halo_code_count);
  d_send.resize(halo_code_count);
  h_send.resize(halo_code_count);
  h_recv.resize(halo_code_count);
  valid.assign(2 * NDIMS, 0);

  const bool need_host_stage = !mpiDefaultMem();

  for (int code = 0; code < halo_code_count; ++code) {
    if (code == halo_center_code) {
      continue;
    }
    int off[NDIMS];
    haloCodeToOffset(code, off);
    if (!haloOffsetNeedsMpi(off, p)) {
      continue;
    }
    active[code] = 1;
    const int64_t hv = haloRegionVolume(off, p);
    halo_vol[code] = hv;
    split_dim[code] = haloRegionSplitDim(off, p);
    const size_t n = static_cast<size_t>(hv * NDIMS * mat_elems);
    d_recv[code] = DeviceView(
        Kokkos::view_alloc("d_recv_halo", Kokkos::WithoutInitializing), n);
    d_send[code] = DeviceView(
        Kokkos::view_alloc("d_send_halo", Kokkos::WithoutInitializing), n);
    if (need_host_stage) {
      h_send[code] = StageView(
          Kokkos::view_alloc("h_send_halo", Kokkos::WithoutInitializing), n);
      h_recv[code] = StageView(
          Kokkos::view_alloc("h_recv_halo", Kokkos::WithoutInitializing), n);
    }
  }
}

template <typename Real> GaugeHaloBuffers<Real>::~GaugeHaloBuffers() {
  // Never leave MPI requests dangling.
  endExchange();
}

//=============================================================================
// Validity tracking
//=============================================================================

template <typename Real> void GaugeHaloBuffers<Real>::invalidate() {
  std::fill(valid.begin(), valid.end(), 0);
}

template <typename Real>
void GaugeHaloBuffers<Real>::markDirty(int dir, int parity) {
  if (dir < 0 || dir >= NDIMS) {
    return;
  }
  if (parity < 0) {
    valid[2 * dir] = 0;
    valid[2 * dir + 1] = 0;
  } else {
    valid[2 * dir + (parity & 1)] = 0;
  }
}

template <typename Real> bool GaugeHaloBuffers<Real>::allValid() const {
  for (char v : valid) {
    if (!v) {
      return false;
    }
  }
  return true;
}

//=============================================================================
// Site lists (boundary / interior) for overlap
//=============================================================================

template <typename Real> void GaugeHaloBuffers<Real>::buildSiteLists() {
  if (site_lists_built) {
    return;
  }
  const LatticeParams &p = this->p;
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
    boundary[parity] = SiteList(
        Kokkos::view_alloc("halo_boundary_sites", Kokkos::WithoutInitializing),
        bnd.size());
    interior[parity] = SiteList(
        Kokkos::view_alloc("halo_interior_sites", Kokkos::WithoutInitializing),
        inr.size());
    Kokkos::View<int64_t *, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        hb(bnd.data(), bnd.size()), hi(inr.data(), inr.size());
    Kokkos::deep_copy(boundary[parity], hb);
    Kokkos::deep_copy(interior[parity], hi);
  }
  site_lists_built = true;
}

template <typename Real>
const typename GaugeHaloBuffers<Real>::SiteList &
GaugeHaloBuffers<Real>::boundarySites(int parity) {
  buildSiteLists();
  return boundary[parity & 1];
}

template <typename Real>
const typename GaugeHaloBuffers<Real>::SiteList &
GaugeHaloBuffers<Real>::interiorSites(int parity) {
  buildSiteLists();
  return interior[parity & 1];
}

//=============================================================================
// Exchange
//=============================================================================

template <typename Real>
void GaugeHaloBuffers<Real>::collectStaleChunks(
    std::vector<Chunk> &chunks) const {
  chunks.clear();
  const int64_t me = mat_elems;
  for (int code = 0; code < halo_code_count; ++code) {
    if (!active[code]) {
      continue;
    }
    const int64_t hv = halo_vol[code];
    const bool split = split_dim[code] >= 0;
    const int64_t block = split ? hv / 2 : hv;
    int nchunk = 0;
    bool open = false;
    int64_t cur_off = 0, cur_n = 0;
    for (int dir = 0; dir < NDIMS; ++dir) {
      const int nblk = split ? 2 : 1;
      for (int b = 0; b < nblk; ++b) {
        const bool stale = split ? !valid[2 * dir + b]
                                 : (!valid[2 * dir] || !valid[2 * dir + 1]);
        const int64_t off =
            (static_cast<int64_t>(dir) * hv + static_cast<int64_t>(b) * block) *
            me;
        const int64_t n = block * me;
        if (stale) {
          if (open && cur_off + cur_n == off) {
            cur_n += n;
          } else {
            if (open) {
              chunks.push_back(
                  Chunk{code, cur_off, cur_n, 2000 + code * 32 + nchunk++});
            }
            cur_off = off;
            cur_n = n;
            open = true;
          }
        }
      }
    }
    if (open) {
      chunks.push_back(
          Chunk{code, cur_off, cur_n, 2000 + code * 32 + nchunk++});
    }
  }
}

template <typename Real>
void GaugeHaloBuffers<Real>::packStale(const ComplexT *gauge_soa,
                                        int64_t soa_stride) {
  const int64_t me = mat_elems;
  const LatticeParams par = p;
  bool launched = false;

  for (int code = 0; code < halo_code_count; ++code) {
    if (!active[code]) {
      continue;
    }
    int off[NDIMS];
    haloCodeToOffset(code, off);
    const int64_t hv = halo_vol[code];
    const int sdim = split_dim[code];
    const bool split = sdim >= 0;

    Kokkos::Array<HaloPackItem, halo_max_pack_items> items;
    int nitems = 0;
    for (int dir = 0; dir < NDIMS; ++dir) {
      if (split) {
        for (int b = 0; b < 2; ++b) {
          if (!valid[2 * dir + b]) {
            items[nitems++] =
                HaloPackItem{dir, static_cast<int64_t>(b) * (hv / 2), hv / 2};
          }
        }
      } else if (!valid[2 * dir] || !valid[2 * dir + 1]) {
        items[nitems++] = HaloPackItem{dir, 0, hv};
      }
    }
    if (nitems == 0) {
      continue;
    }
    // All items of one region have the same size.
    const int64_t item_n = items[0].nslots;
    auto d_send_ptr = d_send[code].data();
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
          haloSlotToCoords(offa.v, sdim, hv, slot, x, par);
          const int64_t idx_eo = coordsToEoIdx(x, par);
          const int64_t base = (static_cast<int64_t>(pi.dir) * hv + slot) * me;
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
void GaugeHaloBuffers<Real>::postChunks(const std::vector<Chunk> &chunks) {
#ifdef KWQFT_USE_MPI
  MPI_Comm comm = kwqftMpiCartComm();
  if (comm == MPI_COMM_NULL) {
    return;
  }
  const LatticeParams &p = this->p;
  int my_coords[NDIMS];
  MPI_Cart_coords(comm, p.rank, NDIMS, my_coords);
  constexpr bool use_default_mem = mpiDefaultMem();

  flight_reqs.clear();
  flight_reqs.reserve(chunks.size() * 2);

  if constexpr (!use_default_mem) {
    for (const auto &c : chunks) {
      const auto rng = std::make_pair(c.elem_off, c.elem_off + c.nelem);
      Kokkos::deep_copy(Kokkos::subview(h_send[c.code], rng),
                        Kokkos::subview(d_send[c.code], rng));
    }
  }

  for (const auto &c : chunks) {
    int off[NDIMS];
    haloCodeToOffset(c.code, off);
    int src_coords[NDIMS], dst_coords[NDIMS];
    for (int d = 0; d < NDIMS; ++d) {
      const int pd = p.proc_grid[d];
      src_coords[d] = ((my_coords[d] + off[d]) % pd + pd) % pd;
      dst_coords[d] = ((my_coords[d] - off[d]) % pd + pd) % pd;
    }
    int src_rank = 0, dst_rank = 0;
    MPI_Cart_rank(comm, src_coords, &src_rank);
    MPI_Cart_rank(comm, dst_coords, &dst_rank);

    const int64_t byte_off =
        c.elem_off * static_cast<int64_t>(sizeof(ComplexT));
    const int nbytes = static_cast<int>(c.nelem * sizeof(ComplexT));

    char *send_base;
    char *recv_base;
    if constexpr (use_default_mem) {
      send_base = reinterpret_cast<char *>(d_send[c.code].data());
      recv_base = reinterpret_cast<char *>(d_recv[c.code].data());
    } else {
      send_base = reinterpret_cast<char *>(h_send[c.code].data());
      recv_base = reinterpret_cast<char *>(h_recv[c.code].data());
    }

    if (src_rank == p.rank && dst_rank == p.rank) {
      // Self-neighbor (cannot happen for active codes, kept for safety).
      if constexpr (use_default_mem) {
        const auto rng = std::make_pair(c.elem_off, c.elem_off + c.nelem);
        Kokkos::deep_copy(Kokkos::subview(d_recv[c.code], rng),
                          Kokkos::subview(d_send[c.code], rng));
      } else {
        std::memcpy(recv_base + byte_off, send_base + byte_off,
                    static_cast<size_t>(nbytes));
      }
      continue;
    }

    MPI_Request rq;
    MPI_Irecv(recv_base + byte_off, nbytes, MPI_BYTE, src_rank, c.tag, comm,
              &rq);
    flight_reqs.push_back(rq);
    MPI_Isend(send_base + byte_off, nbytes, MPI_BYTE, dst_rank, c.tag, comm,
              &rq);
    flight_reqs.push_back(rq);
  }
#else
  (void)chunks;
#endif
}

template <typename Real>
void GaugeHaloBuffers<Real>::beginExchange(const ComplexT *gauge_soa,
                                            int64_t soa_stride, int dir,
                                            int parity) {
  endExchange();
  if (dir >= 0) {
    markDirty(dir, parity);
  }
  if (!p.mpi || p.nproc <= 1) {
    std::fill(valid.begin(), valid.end(), 1);
    return;
  }
  collectStaleChunks(flight_chunks);
  if (flight_chunks.empty()) {
    return;
  }
  packStale(gauge_soa, soa_stride);
  postChunks(flight_chunks);
  in_flight = true;
  // Blocks are valid once endExchange() returns; mark now so that a
  // subsequent markDirty() on another block is not lost.
  std::fill(valid.begin(), valid.end(), 1);
}

template <typename Real> void GaugeHaloBuffers<Real>::endExchange() {
  if (!in_flight) {
    return;
  }
  in_flight = false;
#ifdef KWQFT_USE_MPI
  if (!flight_reqs.empty()) {
    MPI_Waitall(static_cast<int>(flight_reqs.size()), flight_reqs.data(),
                MPI_STATUSES_IGNORE);
    flight_reqs.clear();
  }
  if constexpr (!mpiDefaultMem()) {
    for (const auto &c : flight_chunks) {
      const auto rng = std::make_pair(c.elem_off, c.elem_off + c.nelem);
      Kokkos::deep_copy(Kokkos::subview(d_recv[c.code], rng),
                        Kokkos::subview(h_recv[c.code], rng));
    }
  }
#endif
  // Device-aware MPI completion is not on the Kokkos stream; make the
  // newly arrived ghosts visible to the next kernel. Also waits for any
  // interior kernel that was overlapping the Waitall.
  Kokkos::fence();
  flight_chunks.clear();
}

template <typename Real>
void GaugeHaloBuffers<Real>::refresh(const ComplexT *gauge_soa,
                                     int64_t soa_stride) {
  beginExchange(gauge_soa, soa_stride, -1, -1);
  endExchange();
}

template <typename Real>
GaugeHaloDevice<Real> GaugeHaloBuffers<Real>::deviceView() const {
  GaugeHaloDevice<Real> h;
  for (int code = 0; code < halo_code_count; ++code) {
    if (code == halo_center_code || !active[code]) {
      h.recv[code] = nullptr;
      h.vol[code] = 0;
      h.split_dim[code] = -1;
    } else {
      h.recv[code] = d_recv[code].data();
      h.vol[code] = halo_vol[code];
      h.split_dim[code] = split_dim[code];
    }
  }
  return h;
}

template class GaugeHaloBuffers<double>;
template class GaugeHaloBuffers<float>;

} // namespace kwqft
