/**
 * @file mpi_layout.cpp
 * @brief MPI Cartesian topology matching Chroma-style -geom
 */

#include "mpi_layout.hpp"
#include "constants.hpp"
#include <cstring>
#include <cstdio>
#include <cstdlib>

#ifdef KWQFT_USE_MPI
#include <mpi.h>
#endif

namespace kwqft {

#ifdef KWQFT_USE_MPI
static MPI_Comm g_cart_comm = MPI_COMM_NULL;
MPI_Comm kwqft_mpi_cart_comm() { return g_cart_comm; }
#else
void *kwqft_mpi_cart_comm_opaque() { return nullptr; }
#endif

void mpi_env_init(int *argc, char ***argv) {
#ifdef KWQFT_USE_MPI
  int t = 0;
  MPI_Initialized(&t);
  if (!t) {
    MPI_Init(argc, argv);
  }
#else
  (void)argc;
  (void)argv;
#endif
}

void mpi_env_finalize() {
#ifdef KWQFT_USE_MPI
  int t = 0;
  MPI_Finalized(&t);
  if (!t) {
    MPI_Finalize();
  }
#endif
}

void mpi_setup_cartesian(const int proc_grid[NDIMS], const int global_grid[NDIMS]) {
#ifdef KWQFT_USE_MPI
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int prod = 1;
  for (int d = 0; d < NDIMS; ++d) {
    prod *= proc_grid[d];
  }
  if (prod != size) {
    if (rank == 0) {
      fprintf(stderr,
              "KWQFT: product(proc_grid) = %d must equal MPI size %d\n", prod,
              size);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int d = 0; d < NDIMS; ++d) {
    if (global_grid[d] % proc_grid[d] != 0) {
      if (rank == 0) {
        fprintf(stderr,
                "KWQFT: global_grid[%d]=%d not divisible by proc_grid[%d]=%d\n",
                d, global_grid[d], d, proc_grid[d]);
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  int dims[NDIMS], periods[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    dims[d] = proc_grid[d];
    periods[d] = 1;
  }
  int reorder = 1;
  MPI_Comm old = g_cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &g_cart_comm);
  if (old != MPI_COMM_NULL && old != MPI_COMM_WORLD) {
    MPI_Comm_free(&old);
  }
#else
  (void)proc_grid;
  (void)global_grid;
#endif
}

void mpi_cart_get_coords(int coord[NDIMS]) {
#ifdef KWQFT_USE_MPI
  if (g_cart_comm == MPI_COMM_NULL) {
    for (int d = 0; d < NDIMS; ++d) {
      coord[d] = 0;
    }
    return;
  }
  int rank = 0;
  MPI_Comm_rank(g_cart_comm, &rank);
  MPI_Cart_coords(g_cart_comm, rank, NDIMS, coord);
#else
  for (int d = 0; d < NDIMS; ++d) {
    coord[d] = 0;
  }
#endif
}

int mpi_cart_neighbor(int mu, int sign) {
#ifdef KWQFT_USE_MPI
  if (g_cart_comm == MPI_COMM_NULL) {
    return -1;
  }
  int src = 0, dst = 0;
  MPI_Cart_shift(g_cart_comm, mu, sign, &src, &dst);
  return dst;
#else
  (void)mu;
  (void)sign;
  return -1;
#endif
}

int mpi_comm_rank() {
#ifdef KWQFT_USE_MPI
  int r = 0;
  if (MPI_COMM_WORLD != MPI_COMM_NULL) {
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
  }
  return r;
#else
  return 0;
#endif
}

int mpi_comm_size() {
#ifdef KWQFT_USE_MPI
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
#else
  return 1;
#endif
}

bool parse_geom_argv(int argc, char **argv, int proc_grid[NDIMS],
                     std::vector<std::string> &positional_out) {
  positional_out.clear();
  int start = 1;
  bool found = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-geom") == 0 || std::strcmp(argv[i], "--geom") == 0) {
      if (i + NDIMS >= argc) {
        return false;
      }
      for (int d = 0; d < NDIMS; ++d) {
        proc_grid[d] = std::atoi(argv[i + 1 + d]);
        if (proc_grid[d] <= 0) {
          return false;
        }
      }
      start = i + NDIMS + 1;
      found = true;
      break;
    }
  }
  if (!found) {
    for (int d = 0; d < NDIMS; ++d) {
      proc_grid[d] = 1;
    }
    start = 1;
  }
  for (int i = start; i < argc; ++i) {
    positional_out.push_back(argv[i]);
  }
  return true;
}

} // namespace kwqft
