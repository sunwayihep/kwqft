/**
 * @file heatbath_main.cpp
 * @brief Main program for generating SU(N) gauge configurations using heatbath
 *
 * This is a Kokkos-portable version that can run on CPU or GPU
 * depending on the build configuration
 *
 * CLI (order-independent): \c -geom, \c -latt, \c -beta, \c -ntraj,
 * optional \c -xi0. MPI: \c mpirun -np P with \c ∏ geom_i = P and
 * global \c L_i divisible by \c geom_i (Chroma-style).
 */

#include "io_gauge.hpp"
#include "kwqft.hpp"
#ifdef KWQFT_USE_MPI
#include "gauge_halo.hpp"
#endif
#include "mpi_layout.hpp"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace kwqft;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -latt L0 ... L_{n-1} -beta B -ntraj N [options]\n", prog_name);
  printf("  Options (any order):\n");
  printf("    -geom|--geom p0 ... p_{n-1}   MPI process grid (default 1^%d;\n",
         NDIMS);
  printf("                                   MPI builds only; ∏ p_i = ranks)\n");
  printf("    -latt|--latt L0 ... L_{n-1}    global lattice (required)\n");
  printf("    -beta B                        gauge coupling (required)\n");
  printf("    -ntraj N                       number of trajectories (required)\n");
  printf("    -xi0 X                         bare anisotropy (default 1.0)\n");
  printf("  NDIMS = %d.  -h, --help\n", NDIMS);
  printf("\nExample (serial): %s -latt 8 8 8 16 -beta 6.0 -ntraj 100\n",
         prog_name);
  printf("Example (MPI, 8 ranks): mpirun -np 8 %s -geom 1 2 2 2 -latt 4 4 "
         "4 8 -beta 6.0 -ntraj 10\n",
         prog_name);
}

namespace {

bool parse_heatbath_cli(int argc, char **argv, int proc_grid[NDIMS],
                       std::vector<int> &lattice_size, double &beta, int &ntraj,
                       double &xi0, std::string &err) {
  for (int d = 0; d < NDIMS; ++d) {
    proc_grid[d] = 1;
  }
  xi0 = 1.0;
  lattice_size.assign(NDIMS, 0);
  bool have_geom = false, have_latt = false, have_beta = false;
  bool have_ntraj = false, have_xi0 = false;

  for (int i = 1; i < argc;) {
    const char *a = argv[i];
    if (std::strcmp(a, "-geom") == 0 || std::strcmp(a, "--geom") == 0) {
      if (have_geom) {
        err = "duplicate -geom";
        return false;
      }
      if (i + NDIMS >= argc) {
        err = "-geom must be followed by " + std::to_string(NDIMS) +
              " positive integers";
        return false;
      }
      for (int d = 0; d < NDIMS; ++d) {
        proc_grid[d] = std::atoi(argv[i + 1 + d]);
        if (proc_grid[d] <= 0) {
          err = "invalid -geom: all entries must be positive";
          return false;
        }
      }
      have_geom = true;
      i += 1 + NDIMS;
      continue;
    }
    if (std::strcmp(a, "-latt") == 0 || std::strcmp(a, "--latt") == 0) {
      if (have_latt) {
        err = "duplicate -latt";
        return false;
      }
      if (i + NDIMS >= argc) {
        err = "-latt must be followed by " + std::to_string(NDIMS) +
              " positive integers";
        return false;
      }
      for (int d = 0; d < NDIMS; ++d) {
        lattice_size[static_cast<size_t>(d)] = std::atoi(argv[i + 1 + d]);
        if (lattice_size[static_cast<size_t>(d)] <= 0) {
          err = "invalid -latt: all entries must be positive";
          return false;
        }
      }
      have_latt = true;
      i += 1 + NDIMS;
      continue;
    }
    if (std::strcmp(a, "-beta") == 0) {
      if (have_beta) {
        err = "duplicate -beta";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-beta requires a value";
        return false;
      }
      beta = std::atof(argv[i + 1]);
      have_beta = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-ntraj") == 0) {
      if (have_ntraj) {
        err = "duplicate -ntraj";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-ntraj requires a value";
        return false;
      }
      ntraj = std::atoi(argv[i + 1]);
      have_ntraj = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-xi0") == 0) {
      if (have_xi0) {
        err = "duplicate -xi0";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-xi0 requires a value";
        return false;
      }
      xi0 = std::atof(argv[i + 1]);
      have_xi0 = true;
      i += 2;
      continue;
    }
    err = std::string("unknown or extra argument: ") + a;
    return false;
  }

  if (!have_latt || !have_beta || !have_ntraj) {
    err = "missing required option(s); need -latt, -beta, and -ntraj";
    return false;
  }
  return true;
}

} // namespace

template <typename Real> void run_heatbath(int ntraj) {
  auto &params = PARAMS::params;

#ifdef KWQFT_USE_MPI
  std::unique_ptr<GaugeHaloBuffers<Real>> halo_storage;
  GaugeHaloBuffers<Real> *halo_ptr = nullptr;
  if (params.mpi) {
    halo_storage = std::make_unique<GaugeHaloBuffers<Real>>(params);
    halo_ptr = halo_storage.get();
  }
#endif

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  if (mpi_comm_rank() == 0) {
    gauge.details();
  }

  unsigned int seed = 1234u + static_cast<unsigned int>(mpi_comm_rank());
  RandomGenerator rng(seed, params.half_volume);
  if (mpi_comm_rank() == 0) {
    printf("RNG initialized with seed %u\n", seed);
    printf("Initializing gauge field (cold start)...\n");
  }
  gauge.initCold();

#ifdef KWQFT_USE_MPI
  HeatBath<Real> heatbath(gauge, rng, params, halo_ptr);
  Plaquette<Real> plaquette(gauge, params, halo_ptr);
#else
  HeatBath<Real> heatbath(gauge, rng, params);
  Plaquette<Real> plaquette(gauge, params);
#endif
  Reunitarize<Real> reunitarize(gauge, params);
  PolyakovLoop<Real> polyakov(gauge, params);

  plaquette.run();
  polyakov.run();
  if (mpi_comm_rank() == 0) {
    printf("Initial configuration:\n");
    plaquette.printValue();
    polyakov.printValue();
    printf("\n");
  }

  int num_warmup = 0;
  int save_interval = 10;
  std::ostringstream prefix_stream;
  prefix_stream << "su" << NCOLORS << "_nd" << NDIMS << "_beta" << params.beta;
  for (int i = 0; i < NDIMS; ++i) {
    prefix_stream << "_L" << params.global_grid[i];
  }
  std::string save_prefix = prefix_stream.str();

  Timer total_timer;
  total_timer.start();

  for (int traj = 1; traj <= ntraj; ++traj) {
    if (mpi_comm_rank() == 0) {
      printf("========== Trajectory %d ==========\n", traj);
    }
    Timer traj_timer;
    traj_timer.start();

    heatbath.run();
    reunitarize.run();

    traj_timer.stop();

    plaquette.run();
    polyakov.run();

    if (mpi_comm_rank() == 0) {
      plaquette.printValue();
      polyakov.printValue();
      printf("\nPerformance statistics:\n");
      heatbath.stat();
      reunitarize.stat();
      plaquette.stat();
      polyakov.stat();
      printf("Trajectory time: %.4f s\n\n", traj_timer.elapsed());
    }

    if (traj > num_warmup && traj % save_interval == 0) {
      std::string filename =
          save_prefix + "_cfg_" + std::to_string(traj) + ".bin";
      save_gauge_binary<double, double>(gauge, filename, false);
    }
  }

  total_timer.stop();
  if (mpi_comm_rank() == 0) {
    printf("====================================\n");
    printf("Total simulation time: %.4f s\n", total_timer.elapsed());
    printf("====================================\n");
  }
}

int main(int argc, char *argv[]) {
  kwqft::initialize(argc, argv);

  for (int k = 1; k < argc; ++k) {
    if (std::strcmp(argv[k], "-h") == 0 || std::strcmp(argv[k], "--help") == 0) {
      print_usage(argv[0]);
      kwqft::finalize();
      return 0;
    }
  }

  int proc_grid[NDIMS];
  std::vector<int> lattice_size;
  double beta = 0.0;
  int ntraj = 0;
  double xi0 = 1.0;
  std::string cli_err;
  if (!parse_heatbath_cli(argc, argv, proc_grid, lattice_size, beta, ntraj, xi0,
                          cli_err)) {
    if (mpi_comm_rank() == 0) {
      fprintf(stderr, "Error: %s\n", cli_err.c_str());
      print_usage(argv[0]);
    }
    kwqft::finalize();
    return 1;
  }

  if (beta <= 0.0 || ntraj <= 0 || xi0 <= 0.0) {
    fprintf(stderr, "Error: invalid beta, ntraj, or xi0\n");
    kwqft::finalize();
    return 1;
  }

#ifdef KWQFT_USE_MPI
  int global_lattice[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_lattice[d] = lattice_size[d];
  }
  if (mpi_comm_size() > 1) {
    mpi_setup_cartesian(proc_grid, global_lattice);
    std::vector<int> pg(proc_grid, proc_grid + NDIMS);
    initializeParamsDistributed(lattice_size, pg, beta, true, xi0);
  } else {
    initializeParams(lattice_size, beta, true, xi0);
  }
#else
  initializeParams(lattice_size, beta, true, xi0);
#endif

  if (mpi_comm_rank() == 0) {
    printf("Starting SU(%d) heatbath simulation\n", NCOLORS);
    printf("Beta: %f\n", beta);
    printf("Number of trajectories: %d\n", ntraj);
    printf("Xi0 (bare anisotropy): %f\n", xi0);
    printf("\n");
  }

  run_heatbath<double>(ntraj);

  kwqft::finalize();
  return 0;
}
