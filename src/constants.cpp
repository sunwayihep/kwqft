/**
 * @file constants.cpp
 * @brief Implementation of global lattice parameters
 */

#include "constants.hpp"
#include "mpi_layout.hpp"
#include <cstdio>
#include <memory>

namespace kwqft {

// Global host parameters (plain struct, no Kokkos dependency)
namespace PARAMS {
LatticeParams params;
bool initialized = false;
} // namespace PARAMS

// Use unique_ptr for lazy initialization of Kokkos Views
// This avoids the static initialization order fiasco
static std::unique_ptr<ParamsView> s_device_params;
static std::unique_ptr<ParamsHostView> s_host_params_mirror;

ParamsView &get_device_params() {
  if (!s_device_params) {
    s_device_params = std::make_unique<ParamsView>("device_params");
  }
  return *s_device_params;
}

ParamsHostView &get_host_params_mirror() {
  if (!s_host_params_mirror) {
    s_host_params_mirror = std::make_unique<ParamsHostView>(
        Kokkos::create_mirror_view(get_device_params()));
  }
  return *s_host_params_mirror;
}

void initializeParams(const std::vector<int> &lattice_size, double beta,
                      bool verbose, double xi0) {
  if (PARAMS::initialized) {
    KWQFT_WARNING("Parameters already initialized");
    return;
  }

  PARAMS::params.initialize(lattice_size, beta, xi0);
  PARAMS::initialized = true;

  // Initialize device params view (lazy)
  auto &device_params = get_device_params();
  auto &host_mirror = get_host_params_mirror();

  // Copy to host mirror
  host_mirror() = PARAMS::params;

  // Copy to device
  Kokkos::deep_copy(device_params, host_mirror);

  if (verbose) {
    print_params();
  }
}

void initializeParamsDistributed(const std::vector<int> &global_lattice,
                                 const std::vector<int> &proc_grid, double beta,
                                 bool verbose, double xi0) {
  if (PARAMS::initialized) {
    KWQFT_WARNING("Parameters already initialized");
    return;
  }
  if (static_cast<int>(global_lattice.size()) != NDIMS ||
      static_cast<int>(proc_grid.size()) != NDIMS) {
    KWQFT_ERROR("global_lattice and proc_grid must have NDIMS elements");
  }
  if (global_lattice[0] % proc_grid[0] != 0 ||
      (global_lattice[0] / proc_grid[0]) % 2 != 0) {
    KWQFT_ERROR(
        "Local grid[0] must be even for even/odd ordering (global L0 divisible by 2 * proc_grid[0])");
  }

  LatticeParams &p = PARAMS::params;
  p.mpi = true;
  p.rank = mpi_comm_rank();
  p.nproc = mpi_comm_size();
  mpi_cart_get_coords(p.coord);

  p.volume = 1;
  for (int i = 0; i < NDIMS; ++i) {
    p.global_grid[i] = global_lattice[i];
    p.proc_grid[i] = proc_grid[i];
    if (global_lattice[i] % proc_grid[i] != 0) {
      KWQFT_ERROR("global lattice dimension not divisible by process grid");
    }
    p.grid[i] = global_lattice[i] / proc_grid[i];
    p.grid_with_ghost[i] = p.grid[i];
    p.border[i] = 0;
    p.volume *= static_cast<int64_t>(p.grid[i]);
  }

  p.half_volume = p.volume / 2;
  p.volume_with_ghost = p.volume;
  p.half_volume_with_ghost = p.half_volume;
  p.size = p.volume * NDIMS;

  p.kstride = static_cast<int64_t>(p.grid[0]) * p.grid[1];
  p.tstride = 1;
  for (int i = 0; i < NDIMS - 1; ++i) {
    p.tstride *= static_cast<int64_t>(p.grid[i]);
  }

  p.beta = beta;
  p.beta_over_nc = beta / static_cast<double>(NCOLORS);
  p.xi0 = xi0;

  const int t_dir = NDIMS - 1;
  const double spatial_coeff = 1.0 / xi0;
  const double temporal_coeff = xi0;
  for (int mu = 0; mu < NDIMS; ++mu) {
    for (int nu = 0; nu < NDIMS; ++nu) {
      if (mu == nu) {
        p.coeffs[mu][nu] = 0.0;
      } else if (mu == t_dir || nu == t_dir) {
        p.coeffs[mu][nu] = temporal_coeff;
      } else {
        p.coeffs[mu][nu] = spatial_coeff;
      }
    }
  }

  PARAMS::initialized = true;

  auto &device_params = get_device_params();
  auto &host_mirror = get_host_params_mirror();
  host_mirror() = PARAMS::params;
  Kokkos::deep_copy(device_params, host_mirror);

  if (verbose) {
    print_params();
  }
}

void copy_params_to_device() {
  if (!PARAMS::initialized) {
    KWQFT_ERROR("Parameters not initialized");
    return;
  }

  auto &device_params = get_device_params();
  auto &host_mirror = get_host_params_mirror();

  host_mirror() = PARAMS::params;
  Kokkos::deep_copy(device_params, host_mirror);
}

void print_params() {
  if (PARAMS::params.mpi && mpi_comm_rank() != 0) {
    return;
  }
  printf("==========================================================\n");
  printf("Lattice Parameters:\n");
  if (PARAMS::params.mpi) {
    printf("  Global dimensions: ");
    for (int i = 0; i < NDIMS; ++i) {
      printf("%d", PARAMS::params.global_grid[i]);
      if (i < NDIMS - 1)
        printf(" x ");
    }
    printf("\n");
    printf("  Process grid (-geom): ");
    for (int i = 0; i < NDIMS; ++i) {
      printf("%d", PARAMS::params.proc_grid[i]);
      if (i < NDIMS - 1)
        printf(" x ");
    }
    printf("\n");
    printf("  Local subdomain: ");
  } else {
    printf("  Dimensions: ");
  }
  for (int i = 0; i < NDIMS; ++i) {
    printf("%d", PARAMS::params.grid[i]);
    if (i < NDIMS - 1)
      printf(" x ");
  }
  printf("\n");
  if (PARAMS::params.mpi) {
    int64_t gvol = 1;
    for (int i = 0; i < NDIMS; ++i) {
      gvol *= static_cast<int64_t>(PARAMS::params.global_grid[i]);
    }
    printf("  Global volume: %lld\n", static_cast<long long>(gvol));
    printf("  MPI rank / nproc: %d / %d\n", PARAMS::params.rank,
           PARAMS::params.nproc);
  }
  printf("  Local volume: %lld\n", static_cast<long long>(PARAMS::params.volume));
  printf("  Beta: %.6f\n", PARAMS::params.beta);
  printf("  Beta/Nc: %.6f\n", PARAMS::params.beta_over_nc);
  printf("  Xi0: %.6f\n", PARAMS::params.xi0);
  printf("  Number of colors: %d\n", NCOLORS);
  printf("  Number of dimensions: %d\n", NDIMS);
  printf("==========================================================\n");
}

void finalizeParams() {
  // Release Kokkos views before Kokkos::finalize()
  s_host_params_mirror.reset();
  s_device_params.reset();
  PARAMS::initialized = false;
}

} // namespace kwqft
