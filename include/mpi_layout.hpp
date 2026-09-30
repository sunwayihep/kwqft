/**
 * @file mpi_layout.hpp
 * @brief MPI Cartesian grid (-geom) and environment (implementation in
 * mpi_layout.cpp)
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
/// Cartesian communicator from \ref mpiSetupCartesian (MPI builds only).
MPI_Comm kwqftMpiCartComm(void);
#endif

/// Call once after MPI_Init (no-op if built without MPI).
void mpiEnvInit(int *argc, char ***argv);

/// Call before Kokkos::finalize / process exit (no-op without MPI).
void mpiEnvFinalize();

/**
 * @brief Build NDIMS-dimensional Cartesian communicator; store rank coords.
 *
 * @param proc_grid  p[0]..p[NDIMS-1], product must equal communicator size
 * @param global_grid used for validation (global L[d] divisible by p[d])
 */
void mpiSetupCartesian(const int proc_grid[NDIMS],
                         const int global_grid[NDIMS]);

/// Cartesian coordinates of this rank (after \ref mpi_setup_cartesian).
void mpiCartGetCoords(int coord[NDIMS]);

/// Neighbor rank in direction mu: sign -1 (backward) or +1 (forward). -1 if
/// error.
int mpiCartNeighbor(int mu, int sign);

int mpiCommRank();
int mpiCommSize();

/// Parse "-geom" or "--geom" followed by NDIMS integers; returns true if found.
bool parseGeomArgv(int argc, char **argv, int proc_grid[NDIMS],
                     std::vector<std::string> &positional_out);

} // namespace kwqft

#endif
