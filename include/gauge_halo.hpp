/**
 * @file gauge_halo.hpp
 * @brief MPI face/edge halo buffers for SOA gauge (lazy shift / halo staple)
 *
 * Only regions that cross a domain-decomposed direction are allocated and
 * exchanged. Directions with \c proc_grid[d]==1 wrap locally and never use MPI.
 * Hypercube corners (3+ nonzero offsets) are omitted: Wilson staple/plaquette
 * need at most face+edge ghosts.
 *
 * Buffer layout per region is dir-major and (where possible) parity-split,
 * see \ref halo_region_slot, so one (dir, parity) block is one contiguous
 * message.
 *
 * Validity tracking: the halo remembers which (dir, parity) blocks are up to
 * date. Kernels that modify links call \ref mark_dirty (checkerboard update)
 * or \ref invalidate (whole field); consumers call \ref refresh, which only
 * exchanges stale blocks (no-op if everything is valid). One halo object is
 * shared per \c GaugeArray, so HeatBath / Overrelaxation / Plaquette never
 * re-send data another kernel already sent.
 *
 * Overlap: \ref begin_exchange packs + posts non-blocking MPI and returns;
 * the caller can run interior-site work before \ref end_exchange.
 *
 * With \c KWQFT_MPI_DEVICE_AWARE, MPI uses DefaultMemSpace pointers directly;
 * otherwise non-host-accessible memory is staged through pinned host memory.
 */

#ifndef KWQFT_GAUGE_HALO_HPP
#define KWQFT_GAUGE_HALO_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "gauge_load_save.hpp"
#include "neighbor_access.hpp"
#include "kwqft_common.hpp"
#include <memory>
#include <vector>

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif

namespace kwqft {

/// True when MPI is handed DefaultMemSpace pointers directly (device-aware
/// MPI or host-accessible default memory); false = stage through pinned host.
constexpr bool kwqft_mpi_uses_device_buffers() {
#if defined(KWQFT_MPI_DEVICE_AWARE)
  return true;
#else
  return Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                    DefaultMemSpace>::accessible;
#endif
}

template <typename Real> class GaugeHaloBuffers {
public:
  using ComplexT = Complex<Real>;
  using DeviceView =
      Kokkos::View<ComplexT *, Kokkos::LayoutRight, DefaultMemSpace>;
  using StageView = Kokkos::View<ComplexT *, Kokkos::SharedHostPinnedSpace>;
  using SiteList = Kokkos::View<int64_t *, DefaultMemSpace>;

  explicit GaugeHaloBuffers(const LatticeParams &p);
  ~GaugeHaloBuffers();

  GaugeHaloBuffers(const GaugeHaloBuffers &) = delete;
  GaugeHaloBuffers &operator=(const GaugeHaloBuffers &) = delete;

  //--- validity tracking -----------------------------------------------------

  /// Mark every (dir, parity) block stale (whole field was rewritten).
  void invalidate();

  /// Mark links of direction \p dir on parity \p parity stale.
  void mark_dirty(int dir, int parity);

  bool all_valid() const;

  //--- exchange --------------------------------------------------------------

  /// Exchange all stale blocks synchronously (no-op if all valid).
  void refresh(const ComplexT *gauge_soa, int64_t soa_stride);

  /**
   * @brief Mark (dir, parity) stale, pack all stale blocks and post MPI.
   *
   * Returns as soon as sends/receives are posted. Ghost buffers must not be
   * read until \ref end_exchange.  If \p dir < 0 no block is marked, only the
   * currently stale ones are sent.
   */
  void begin_exchange(const ComplexT *gauge_soa, int64_t soa_stride,
                      int dir = -1, int parity = -1);

  /// Complete an exchange started by \ref begin_exchange (no-op otherwise).
  void end_exchange();

  bool in_flight() const { return in_flight_; }

  //--- site lists for compute/communication overlap ---------------------------

  /// Half-volume ids (parity \p parity) of sites on some exchanged face.
  const SiteList &boundary_sites(int parity);
  /// Half-volume ids (parity \p parity) whose staple never reads a ghost.
  const SiteList &interior_sites(int parity);

  //--- device access ---------------------------------------------------------

  GaugeHaloDevice<Real> device_view() const;

  /**
   * @brief Apply a per-matrix functor \c f(MatrixSun&) to every ghost link.
   *
   * Used to keep ghosts valid through deterministic link-wise operations
   * (e.g. reunitarization) without a new exchange.
   */
  template <class F> void apply_to_ghosts(F f) {
    end_exchange();
    const int64_t me = static_cast<int64_t>(NCOLORS * NCOLORS);
    for (int code = 0; code < HALO_CODE_COUNT; ++code) {
      if (!active_[code]) {
        continue;
      }
      auto buf = d_recv_[code];
      const int64_t nmat = static_cast<int64_t>(buf.extent(0)) / me;
      Kokkos::parallel_for(
          "halo_apply_to_ghosts",
          Kokkos::RangePolicy<DefaultExecSpace>(0, nmat),
          KOKKOS_LAMBDA(const int64_t m) {
            MatrixSun<Real, NCOLORS> U;
            ComplexT *ptr = buf.data() + m * me;
            for (int i = 0; i < NCOLORS; ++i) {
              for (int j = 0; j < NCOLORS; ++j) {
                U.e[i][j] = ptr[j + i * NCOLORS];
              }
            }
            f(U);
            for (int i = 0; i < NCOLORS; ++i) {
              for (int j = 0; j < NCOLORS; ++j) {
                ptr[j + i * NCOLORS] = U.e[i][j];
              }
            }
          });
    }
    Kokkos::fence();
  }

  const LatticeParams &params() const { return p_; }

private:
  static constexpr bool mpi_default_mem() {
    return kwqft_mpi_uses_device_buffers();
  }

  /// Contiguous element range of one region buffer to exchange.
  struct Chunk {
    int code;
    int64_t elem_off;
    int64_t nelem;
    int tag;
  };

public:
  // Implementation helpers (public only because nvcc requires the enclosing
  // function of an extended lambda to be publicly accessible).
  void pack_stale(const ComplexT *gauge_soa, int64_t soa_stride);

private:
  void build_site_lists();
  void collect_stale_chunks(std::vector<Chunk> &chunks) const;
  void post_chunks(const std::vector<Chunk> &chunks);

  LatticeParams p_{};
  int64_t mat_elems_{0};
  std::vector<int64_t> halo_vol_;
  std::vector<int> split_dim_;
  std::vector<char> active_;
  std::vector<DeviceView> d_recv_;
  std::vector<DeviceView> d_send_;
  std::vector<StageView> h_send_;
  std::vector<StageView> h_recv_;

  /// valid_[dir * 2 + parity]
  std::vector<char> valid_;

  bool in_flight_{false};
  std::vector<Chunk> flight_chunks_;
#ifdef KWQFT_USE_MPI
  std::vector<MPI_Request> flight_reqs_;
#endif

  bool site_lists_built_{false};
  SiteList boundary_[2];
  SiteList interior_[2];
};

} // namespace kwqft

#endif
