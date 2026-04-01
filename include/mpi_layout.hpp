/**
 * @file mpi_layout.hpp
 * @brief MPI Cartesian grid (-geom) and environment (implementation in mpi_layout.cpp)
 */

#ifndef KWQFT_MPI_LAYOUT_HPP
#define KWQFT_MPI_LAYOUT_HPP

#include <string>
#include <vector>

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif

namespace kwqft {

#ifdef KWQFT_USE_MPI
/// Cartesian communicator from \ref mpi_setup_cartesian (MPI builds only).
MPI_Comm kwqft_mpi_cart_comm(void);
#endif

/// Call once after MPI_Init (no-op if built without MPI).
void mpi_env_init(int *argc, char ***argv);

/// Call before Kokkos::finalize / process exit (no-op without MPI).
void mpi_env_finalize();

/**
 * @brief Build NDIMS-dimensional Cartesian communicator; store rank coords.
 *
 * @param proc_grid  p[0]..p[NDIMS-1], product must equal communicator size
 * @param global_grid used for validation (global L[d] divisible by p[d])
 */
void mpi_setup_cartesian(const int proc_grid[NDIMS],
                         const int global_grid[NDIMS]);

/// Cartesian coordinates of this rank (after \ref mpi_setup_cartesian).
void mpi_cart_get_coords(int coord[NDIMS]);

/// Neighbor rank in direction mu: sign -1 (backward) or +1 (forward). -1 if error.
int mpi_cart_neighbor(int mu, int sign);

int mpi_comm_rank();
int mpi_comm_size();

/// Parse "-geom" or "--geom" followed by NDIMS integers; returns true if found.
bool parse_geom_argv(int argc, char **argv, int proc_grid[NDIMS],
                     std::vector<std::string> &positional_out);

} // namespace kwqft

#endif
