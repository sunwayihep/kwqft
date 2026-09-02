/**
 * @file shift_layout.hpp
 * @brief Field layouts for generic \ref ShiftMap / \c shift()
 *
 * QDPXX-style \c shift(field, dir, mu) is field-type agnostic; only the
 * per-site storage layout differs. Add new \ref ShiftLayoutKind values (e.g.
 * fermion SOA) without changing the MPI shift engine.
 */

#ifndef KWQFT_SHIFT_LAYOUT_HPP
#define KWQFT_SHIFT_LAYOUT_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "kwqft_common.hpp"

namespace kwqft {

enum ShiftLayoutKind {
  SHIFT_LAYOUT_DENSE_EO = 0,
  SHIFT_LAYOUT_GAUGE_SOA = 1,
  // SHIFT_LAYOUT_FERMION_SOA = 2,  // future
};

/// Describes how to read/write one lattice site's degrees of freedom.
template <typename Real> struct ShiftLayout {
  ShiftLayoutKind kind{SHIFT_LAYOUT_DENSE_EO};
  int site_elems{0};
  int64_t stride{0};
  int component{0};
  int64_t volume{0};

  KOKKOS_INLINE_FUNCTION
  static ShiftLayout dense(int site_elems) {
    ShiftLayout l;
    l.kind = SHIFT_LAYOUT_DENSE_EO;
    l.site_elems = site_elems;
    return l;
  }

  KOKKOS_INLINE_FUNCTION
  static ShiftLayout gauge_soa(int64_t soa_stride, int link_dir, int64_t volume) {
    ShiftLayout l;
    l.kind = SHIFT_LAYOUT_GAUGE_SOA;
    l.site_elems = NCOLORS * NCOLORS;
    l.stride = soa_stride;
    l.component = link_dir;
    l.volume = volume;
    return l;
  }
};

/// Load one site's values from a dense EO shift buffer.
template <typename Real>
KOKKOS_INLINE_FUNCTION void loadDenseSite(const Complex<Real> *dense,
                                           int64_t idx_eo, int site_elems,
                                           Complex<Real> *out) {
  const int64_t base = idx_eo * static_cast<int64_t>(site_elems);
  for (int k = 0; k < site_elems; ++k) {
    out[k] = dense[base + k];
  }
}

/// Copy one site between arbitrary layouts (device-callable).
template <typename Real>
KOKKOS_INLINE_FUNCTION void shift_copy_site(const ShiftLayout<Real> &layout,
                                            const Complex<Real> *src,
                                            Complex<Real> *dst, int64_t dst_idx,
                                            int64_t src_idx) {
  switch (layout.kind) {
  case SHIFT_LAYOUT_DENSE_EO: {
    const int64_t base_dst = dst_idx * static_cast<int64_t>(layout.site_elems);
    const int64_t base_src = src_idx * static_cast<int64_t>(layout.site_elems);
    for (int k = 0; k < layout.site_elems; ++k) {
      dst[base_dst + k] = src[base_src + k];
    }
    break;
  }
  case SHIFT_LAYOUT_GAUGE_SOA: {
    const int64_t base_src =
        src_idx + static_cast<int64_t>(layout.component) * layout.volume;
    const int64_t base_dst = dst_idx * static_cast<int64_t>(layout.site_elems);
    for (int k = 0; k < layout.site_elems; ++k) {
      dst[base_dst + k] = src[base_src + static_cast<int64_t>(k) * layout.stride];
    }
    break;
  }
  default:
    break;
  }
}

/// Read one site into a contiguous buffer \p out[0..site_elems).
template <typename Real>
KOKKOS_INLINE_FUNCTION void shift_read_site(const ShiftLayout<Real> &layout,
                                            const Complex<Real> *src,
                                            int64_t src_idx, Complex<Real> *out) {
  switch (layout.kind) {
  case SHIFT_LAYOUT_DENSE_EO: {
    const int64_t base_src = src_idx * static_cast<int64_t>(layout.site_elems);
    for (int k = 0; k < layout.site_elems; ++k) {
      out[k] = src[base_src + k];
    }
    break;
  }
  case SHIFT_LAYOUT_GAUGE_SOA: {
    const int64_t base_src =
        src_idx + static_cast<int64_t>(layout.component) * layout.volume;
    for (int k = 0; k < layout.site_elems; ++k) {
      out[k] = src[base_src + static_cast<int64_t>(k) * layout.stride];
    }
    break;
  }
  default:
    break;
  }
}

} // namespace kwqft

#endif
