/**
 * @file constants.hpp
 * @brief Global lattice parameters and constants for KWQFT
 *
 * Provides host and device accessible lattice parameters
 */

#ifndef KWQFT_CONSTANTS_HPP
#define KWQFT_CONSTANTS_HPP

#include "kwqft_common.hpp"
#include <array>
#include <vector>

namespace kwqft {

/**
 * @brief Structure to hold lattice parameters
 * This is designed to be copyable to device
 * Uses int64_t for volume-related fields to support large lattices
 */
struct LatticeParams {
  int grid[NDIMS];            // Lattice dimensions (per-dimension, int is enough)
  int grid_with_ghost[NDIMS]; // Lattice dimensions including ghost zones
  int border[NDIMS];          // Border size for multi-GPU

  // Volume-related fields use int64_t to support high-dimensional lattices
  int64_t volume;                 // Total volume
  int64_t half_volume;            // Half volume (for even/odd)
  int64_t volume_with_ghost;      // Volume including ghost zones
  int64_t half_volume_with_ghost;
  int64_t size;    // Total number of links = volume * NDIMS
  int64_t kstride; // Stride for k = nx * ny
  int64_t tstride; // Stride for t = nx * ny * nz (or product of first NDIMS-1 dims)

  double beta;         // Gauge coupling
  double beta_over_nc; // beta / Nc

  bool use_texture; // Use texture memory (for CUDA)

  // Default constructor
  KOKKOS_INLINE_FUNCTION
  LatticeParams()
      : volume(0), half_volume(0), volume_with_ghost(0),
        half_volume_with_ghost(0), size(0), kstride(0), tstride(0), beta(0.0),
        beta_over_nc(0.0), use_texture(false) {
    for (int i = 0; i < NDIMS; ++i) {
      grid[i] = 0;
      grid_with_ghost[i] = 0;
      border[i] = 0;
    }
  }

  // Initialize from lattice dimensions and beta
  void initialize(const std::vector<int> &lattice_size, double _beta) {
    if (static_cast<int>(lattice_size.size()) != NDIMS) {
      KWQFT_ERROR("Lattice size vector must have NDIMS elements");
    }

    volume = 1;
    for (int i = 0; i < NDIMS; ++i) {
      grid[i] = lattice_size[i];
      grid_with_ghost[i] = lattice_size[i];
      border[i] = 0;
      volume *= static_cast<int64_t>(grid[i]);
    }

    half_volume = volume / 2;
    volume_with_ghost = volume;
    half_volume_with_ghost = half_volume;
    size = volume * NDIMS;

    // Compute strides for spatial volume
    kstride = static_cast<int64_t>(grid[0]) * grid[1];
    tstride = 1;
    for (int i = 0; i < NDIMS - 1; ++i) {
      tstride *= static_cast<int64_t>(grid[i]);
    }

    beta = _beta;
    beta_over_nc = beta / static_cast<double>(NCOLORS);
  }

  // Get grid dimension
  KOKKOS_INLINE_FUNCTION
  int get_grid(int dim) const { return grid[dim]; }

  // Get grid dimension with ghost
  KOKKOS_INLINE_FUNCTION
  int get_grid_g(int dim) const { return grid_with_ghost[dim]; }

  // Get border
  KOKKOS_INLINE_FUNCTION
  int get_border(int dim) const { return border[dim]; }
};

// Global host parameters (to be initialized at startup)
namespace PARAMS {
extern LatticeParams params;
extern bool initialized;
} // namespace PARAMS

// Note: Kokkos Views cannot be global variables because they require
// Kokkos::initialize() to be called first. We use pointers with lazy init.
using ParamsView = Kokkos::View<LatticeParams, DefaultMemSpace>;
using ParamsHostView = typename ParamsView::HostMirror;

/**
 * @brief Get the device parameters view (lazy initialization)
 */
ParamsView &get_device_params();

/**
 * @brief Get the host mirror view (lazy initialization)
 */
ParamsHostView &get_host_params_mirror();

/**
 * @brief Initialize the global lattice parameters
 */
void initializeParams(const std::vector<int> &lattice_size, double beta,
                      bool verbose = true);

/**
 * @brief Copy parameters to device memory
 */
void copy_params_to_device();

/**
 * @brief Print lattice details
 */
void print_params();

/**
 * @brief Cleanup Kokkos views (call before Kokkos::finalize)
 */
void finalizeParams();

//=============================================================================
// Inline accessor functions that work on both host and device
//=============================================================================

KOKKOS_INLINE_FUNCTION
int param_Grid(const LatticeParams &p, int dim) { return p.grid[dim]; }

KOKKOS_INLINE_FUNCTION
int param_GridG(const LatticeParams &p, int dim) {
  return p.grid_with_ghost[dim];
}

KOKKOS_INLINE_FUNCTION
int64_t param_Volume(const LatticeParams &p) { return p.volume; }

KOKKOS_INLINE_FUNCTION
int64_t param_HalfVolume(const LatticeParams &p) { return p.half_volume; }

KOKKOS_INLINE_FUNCTION
int64_t param_VolumeG(const LatticeParams &p) { return p.volume_with_ghost; }

KOKKOS_INLINE_FUNCTION
int64_t param_HalfVolumeG(const LatticeParams &p) {
  return p.half_volume_with_ghost;
}

KOKKOS_INLINE_FUNCTION
int64_t param_Size(const LatticeParams &p) { return p.size; }

KOKKOS_INLINE_FUNCTION
double param_Beta(const LatticeParams &p) { return p.beta; }

KOKKOS_INLINE_FUNCTION
double param_BetaOverNc(const LatticeParams &p) { return p.beta_over_nc; }

KOKKOS_INLINE_FUNCTION
int param_border(const LatticeParams &p, int dim) { return p.border[dim]; }

} // namespace kwqft

#endif // KWQFT_CONSTANTS_HPP
