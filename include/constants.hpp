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
  int grid[NDIMS]; // Lattice dimensions (per-dimension, int is enough)
  int grid_with_ghost[NDIMS]; // Lattice dimensions including ghost zones
  int border[NDIMS];          // Border size for multi-GPU

  // Volume-related fields use int64_t to support high-dimensional lattices
  int64_t volume;            // Total volume
  int64_t half_volume;       // Half volume (for even/odd)
  int64_t volume_with_ghost; // Volume including ghost zones
  int64_t half_volume_with_ghost;
  int64_t size;    // Total number of links = volume * NDIMS
  int64_t kstride; // Stride for k = nx * ny
  int64_t
      tstride; // Stride for t = nx * ny * nz (or product of first NDIMS-1 dims)

  double beta;                 // Gauge coupling
  double beta_over_nc;         // beta / Nc
  double xi0;                  // Bare anisotropy, xi0=1 for isotropic
  double coeffs[NDIMS][NDIMS]; // Anisotropic plaquette coefficients

  bool use_texture; // Use texture memory (for CUDA)

  /// Domain decomposition (MPI): when false, grid[] is the full lattice
  /// (periodic).
  bool mpi{false};
  int global_grid[NDIMS]; // Global lattice (user-specified sizes)
  int proc_grid[NDIMS];   // MPI process grid (1 if not using MPI)
  int coord[NDIMS];       // This rank's Cartesian coordinates
  int rank{0};
  int nproc{1};

  // Default constructor
  KOKKOS_INLINE_FUNCTION
  LatticeParams()
      : volume(0), half_volume(0), volume_with_ghost(0),
        half_volume_with_ghost(0), size(0), kstride(0), tstride(0), beta(0.0),
        beta_over_nc(0.0), xi0(1.0), use_texture(false) {
    for (int i = 0; i < NDIMS; ++i) {
      grid[i] = 0;
      grid_with_ghost[i] = 0;
      border[i] = 0;
      global_grid[i] = 0;
      proc_grid[i] = 1;
      coord[i] = 0;
      for (int j = 0; j < NDIMS; ++j) {
        coeffs[i][j] = (i == j) ? 0.0 : 1.0;
      }
    }
  }

  // Initialize from lattice dimensions and beta
  void initialize(const std::vector<int> &lattice_size, double beta,
                  double xi0 = 1.0) {
    if (static_cast<int>(lattice_size.size()) != NDIMS) {
      KWQFT_ERROR("Lattice size vector must have NDIMS elements");
    }
    if (lattice_size[0] % 2 != 0) {
      KWQFT_ERROR("First lattice dimension (grid[0]) must be even for even/odd "
                  "ordering");
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

    this->beta = beta;
    beta_over_nc = beta / static_cast<double>(NCOLORS);
    this->xi0 = xi0;
    mpi = false;
    rank = 0;
    nproc = 1;
    for (int i = 0; i < NDIMS; ++i) {
      global_grid[i] = grid[i];
      proc_grid[i] = 1;
      coord[i] = 0;
    }

    // Wilson anisotropic plaquette convention:
    // spatial-spatial: beta/xi0, spatial-temporal: beta*xi0.
    // The global beta/Nc factor is applied in heatbath, so here we only store
    // direction-dependent multipliers.
    const int t_dir = NDIMS - 1;
    const double spatial_coeff = 1.0 / xi0;
    const double temporal_coeff = xi0;
    for (int mu = 0; mu < NDIMS; ++mu) {
      for (int nu = 0; nu < NDIMS; ++nu) {
        if (mu == nu) {
          coeffs[mu][nu] = 0.0;
        } else if (mu == t_dir || nu == t_dir) {
          coeffs[mu][nu] = temporal_coeff;
        } else {
          coeffs[mu][nu] = spatial_coeff;
        }
      }
    }
  }

  // Get grid dimension
  KOKKOS_INLINE_FUNCTION
  int getGrid(int dim) const { return grid[dim]; }

  // Get grid dimension with ghost
  KOKKOS_INLINE_FUNCTION
  int getGridG(int dim) const { return grid_with_ghost[dim]; }

  // Get border
  KOKKOS_INLINE_FUNCTION
  int getBorder(int dim) const { return border[dim]; }
};

// Global host parameters (to be initialized at startup)
namespace PARAMS {
extern LatticeParams params;
extern bool initialized;
} // namespace PARAMS

// Note: Kokkos Views cannot be global variables because they require
// Kokkos::initialize() to be called first. We use pointers with lazy init.
using ParamsView = Kokkos::View<LatticeParams, DefaultMemSpace>;
using ParamsHostView = typename ParamsView::host_mirror_type;

/**
 * @brief Get the device parameters view (lazy initialization)
 */
ParamsView &getDeviceParams();

/**
 * @brief Get the host mirror view (lazy initialization)
 */
ParamsHostView &getHostParamsMirror();

/**
 * @brief Initialize the global lattice parameters
 */
void initializeParams(const std::vector<int> &lattice_size, double beta,
                      bool verbose = true, double xi0 = 1.0);

/**
 * @brief Initialize parameters for MPI domain decomposition.
 *
 * @param global_lattice Full lattice sizes L[0]..L[NDIMS-1]
 * @param proc_grid      Process counts p[0]..p[NDIMS-1] (product = MPI size)
 * @param beta           Coupling
 * @param verbose        Print parameters on rank 0
 * @param xi0            Anisotropy (same as \ref initializeParams)
 */
void initializeParamsDistributed(const std::vector<int> &global_lattice,
                                 const std::vector<int> &proc_grid, double beta,
                                 bool verbose = true, double xi0 = 1.0);

/**
 * @brief Copy parameters to device memory
 */
void copyParamsToDevice();

/**
 * @brief Print lattice details
 */
void printParams();

/**
 * @brief Cleanup Kokkos views (call before Kokkos::finalize)
 */
void finalizeParams();

//=============================================================================
// Inline accessor functions that work on both host and device
//=============================================================================

KOKKOS_INLINE_FUNCTION
int paramGrid(const LatticeParams &p, int dim) { return p.grid[dim]; }

KOKKOS_INLINE_FUNCTION
int paramGridG(const LatticeParams &p, int dim) {
  return p.grid_with_ghost[dim];
}

KOKKOS_INLINE_FUNCTION
int64_t paramVolume(const LatticeParams &p) { return p.volume; }

KOKKOS_INLINE_FUNCTION
int64_t paramHalfVolume(const LatticeParams &p) { return p.half_volume; }

KOKKOS_INLINE_FUNCTION
int64_t paramVolumeG(const LatticeParams &p) { return p.volume_with_ghost; }

KOKKOS_INLINE_FUNCTION
int64_t paramHalfVolumeG(const LatticeParams &p) {
  return p.half_volume_with_ghost;
}

KOKKOS_INLINE_FUNCTION
int64_t paramSize(const LatticeParams &p) { return p.size; }

KOKKOS_INLINE_FUNCTION
double paramBeta(const LatticeParams &p) { return p.beta; }

KOKKOS_INLINE_FUNCTION
double paramBetaOverNc(const LatticeParams &p) { return p.beta_over_nc; }

KOKKOS_INLINE_FUNCTION
int paramBorder(const LatticeParams &p, int dim) { return p.border[dim]; }

} // namespace kwqft

#endif // KWQFT_CONSTANTS_HPP
