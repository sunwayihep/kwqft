/**
 * @file shift_map.cpp
 * @brief QDPXX Map-style nearest-neighbor shift (MPI via face Sendrecv)
 */

#include "constants.hpp"
#include "shift_map.hpp"
#include "index.hpp"

#include <memory>

#ifdef KWQFT_USE_MPI
#include "mpi_layout.hpp"
#include <mpi.h>
#endif

namespace kwqft {

namespace {

template <typename Real> struct ShiftMapHolder {
  static std::unique_ptr<ShiftMap<Real>> instance;
};

template <typename Real>
std::unique_ptr<ShiftMap<Real>> ShiftMapHolder<Real>::instance;

} // namespace

template <typename Real> ShiftMap<Real> &shiftMap() {
  if (!ShiftMapHolder<Real>::instance) {
    KWQFT_ERROR("ShiftMap not initialized (call initializeParams first)");
  }
  return *ShiftMapHolder<Real>::instance;
}

template <typename Real> void initializeShiftMap(const LatticeParams &p) {
  ShiftMapHolder<Real>::instance = std::make_unique<ShiftMap<Real>>(p);
}

template <typename Real> void finalizeShiftMap() {
  ShiftMapHolder<Real>::instance.reset();
}

template ShiftMap<double> &shiftMap<double>();
template ShiftMap<float> &shiftMap<float>();
template void initializeShiftMap<double>(const LatticeParams &p);
template void initializeShiftMap<float>(const LatticeParams &p);
template void finalizeShiftMap<double>();
template void finalizeShiftMap<float>();

namespace {

KOKKOS_INLINE_FUNCTION int64_t tangential_face_index(const int x[NDIMS], int mu,
                                                     const LatticeParams &p) {
  int64_t idx = 0;
  int64_t mult = 1;
  for (int d = 0; d < NDIMS; ++d) {
    if (d == mu) {
      continue;
    }
    idx += static_cast<int64_t>(x[d]) * mult;
    mult *= static_cast<int64_t>(p.grid[d]);
  }
  return idx;
}

KOKKOS_INLINE_FUNCTION void face_coords(int fidx, int mu, int face_val,
                                        int x[NDIMS], const LatticeParams &p) {
  int64_t t = fidx;
  for (int d = 0; d < NDIMS; ++d) {
    if (d == mu) {
      x[d] = face_val;
    } else {
      x[d] = static_cast<int>(t % static_cast<int64_t>(p.grid[d]));
      t /= static_cast<int64_t>(p.grid[d]);
    }
  }
}

} // namespace

template <typename Real>
ShiftMap<Real>::ShiftMap(const LatticeParams &p) : p_(p) {
  for (int mu = 0; mu < NDIMS; ++mu) {
    face_vol_[mu] = shift_face_volume(mu, p);
  }
  h_send_.resize(NDIMS);
  h_recv_.resize(NDIMS);
  d_ghost_.resize(NDIMS);
}

template <typename Real>
void ShiftMap<Real>::prepare_site_elems(int site_elems) {
  if (site_elems <= cached_site_elems_) {
    return;
  }
  cached_site_elems_ = site_elems;
  const int64_t dense_n = p_.volume * static_cast<int64_t>(site_elems);
  for (int i = 0; i < SHIFT_BUF_COUNT; ++i) {
    d_dense_[i] = Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>(
        Kokkos::view_alloc("shift_dense", Kokkos::WithoutInitializing),
        static_cast<size_t>(dense_n));
  }
  for (int mu = 0; mu < NDIMS; ++mu) {
    const size_t fn = static_cast<size_t>(face_vol_[mu] *
                                          static_cast<int64_t>(site_elems));
    h_send_[mu] = Kokkos::View<ComplexT *, Kokkos::HostSpace>(
        Kokkos::view_alloc("shift_h_send", Kokkos::WithoutInitializing), fn);
    h_recv_[mu] = Kokkos::View<ComplexT *, Kokkos::HostSpace>(
        Kokkos::view_alloc("shift_h_recv", Kokkos::WithoutInitializing), fn);
    d_ghost_[mu] = Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>(
        Kokkos::view_alloc("shift_d_ghost", Kokkos::WithoutInitializing), fn);
  }
}

template <typename Real>
Complex<Real> *ShiftMap<Real>::alloc_dense_buf(int site_elems) {
  if (buf_next_ >= SHIFT_BUF_COUNT) {
    KWQFT_ERROR("ShiftMap: out of shift buffers (increase SHIFT_BUF_COUNT)");
  }
  prepare_site_elems(site_elems);
  ComplexT *ptr = d_dense_[buf_next_].data();
  buf_next_++;
  return ptr;
}

template <typename Real>
void ShiftMap<Real>::shift_impl(const ComplexT *src, ComplexT *dst,
                                const Layout &layout, int shift_mu, int isign) {
  const LatticeParams par = p_;
  const int site_elems = layout.site_elems;
  const int64_t vol = par.volume;
  const int mu = shift_mu;
  const ComplexT *src_ptr = src;

  const bool do_mpi =
#ifdef KWQFT_USE_MPI
      par.mpi && par.nproc > 1 && par.proc_grid[mu] > 1;
#else
      false;
#endif
  ComplexT *ghost_ptr = nullptr;

#ifdef KWQFT_USE_MPI
  if (do_mpi) {
    const int64_t fv = face_vol_[mu];
    const int nbytes =
        static_cast<int>(fv * static_cast<int64_t>(site_elems) * sizeof(ComplexT));
    auto d_pack = d_ghost_[mu];
    auto d_pack_ptr = d_pack.data();
    auto h_send = h_send_[mu];
    auto h_recv = h_recv_[mu];
    auto d_ghost_v = d_ghost_[mu];
    MPI_Comm comm = kwqft_mpi_cart_comm();

    const int send_face =
        (isign == FORWARD) ? 0 : par.grid[mu] - 1;
    const int peer = (isign == FORWARD) ? mpi_cart_neighbor(mu, +1)
                                              : mpi_cart_neighbor(mu, -1);
    const int tag = 3000 + mu * 2 + (isign == FORWARD ? 0 : 1);

    if (peer >= 0) {
      Kokkos::parallel_for(
          "shift_pack_face",
          Kokkos::RangePolicy<DefaultExecSpace>(0, fv),
          KOKKOS_LAMBDA(const int64_t fidx) {
            int x[NDIMS];
            face_coords(fidx, mu, send_face, x, par);
            const int64_t idx_eo = coords_to_eo_idx(x, par);
            const int64_t base_dst = fidx * static_cast<int64_t>(site_elems);
            shift_read_site(layout, src_ptr, idx_eo, d_pack_ptr + base_dst);
          });
      Kokkos::fence();
      Kokkos::deep_copy(h_send, d_pack);

      MPI_Sendrecv(h_send.data(), nbytes, MPI_BYTE, peer, tag, h_recv.data(),
                   nbytes, MPI_BYTE, peer, tag, comm, MPI_STATUS_IGNORE);
      Kokkos::deep_copy(d_ghost_v, h_recv);
      ghost_ptr = d_ghost_v.data();
    }
  }
#endif

  Kokkos::parallel_for(
      "shift_field",
      Kokkos::RangePolicy<DefaultExecSpace>(0, vol),
      KOKKOS_LAMBDA(const int64_t idx_eo) {
        int x[NDIMS];
        const int oddbit = (idx_eo >= par.half_volume) ? 1 : 0;
        const int64_t id =
            idx_eo - static_cast<int64_t>(oddbit) * par.half_volume;
        indexNdEo(x, id, oddbit, par);

        const bool need_ghost =
            do_mpi &&
            ((isign == FORWARD && x[mu] == par.grid[mu] - 1) ||
             (isign == BACKWARD && x[mu] == 0));

        if (!need_ghost) {
          const int64_t src_idx = shift_eo(idx_eo, mu, isign, par);
          shift_copy_site(layout, src_ptr, dst, idx_eo, src_idx);
        } else {
          const int64_t fidx = tangential_face_index(x, mu, par);
          const int64_t base_ghost = fidx * static_cast<int64_t>(site_elems);
          const int64_t base_dst = idx_eo * static_cast<int64_t>(site_elems);
          for (int k = 0; k < site_elems; ++k) {
            dst[base_dst + k] = ghost_ptr[base_ghost + k];
          }
        }
      });
  Kokkos::fence();
}

template <typename Real>
const Complex<Real> *ShiftMap<Real>::shift(const ComplexT *field,
                                           const Layout &layout, int isign,
                                           int mu) {
  ComplexT *dst = alloc_dense_buf(layout.site_elems);
  shift_impl(field, dst, layout, mu, isign);
  return dst;
}

template class ShiftMap<double>;
template class ShiftMap<float>;

} // namespace kwqft
