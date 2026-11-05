/**
 * @file heatbath_main.cpp
 * @brief Main program for generating SU(N) gauge configurations using heatbath
 *
 * This is a Kokkos-portable version that can run on CPU or GPU
 * depending on the build configuration
 *
 * CLI (order-independent): \c -geom, \c -latt, \c -beta, \c -ntraj,
 * optional \c -xi0, \c -nhb, \c -novr, \c -nsave, \c -seed, \c -start,
 * \c -traj-offset, and \c -warmup. MPI: \c mpirun -np P with
 * \c ∏ geom_i = P and global \c L_i divisible by \c geom_i (Chroma-style).
 *
 * One trajectory = \c nhb pseudo-heatbath sweeps followed by \c novr
 * overrelaxation sweeps, then reunitarize and measure.
 */

#include "io_gauge.hpp"
#include "kwqft.hpp"
#include "mpi_layout.hpp"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifndef KWQFT_REAL_TYPE
#define KWQFT_REAL_TYPE double
#endif

using namespace kwqft;

void printUsage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -latt L0 ... L_{n-1} -beta B -ntraj N [options]\n", prog_name);
  printf("  Options (any order):\n");
  printf("    -geom|--geom p0 ... p_{n-1}   MPI process grid (default 1^%d;\n",
         NDIMS);
  printf(
      "                                   MPI builds only; ∏ p_i = ranks)\n");
  printf("    -latt|--latt L0 ... L_{n-1}    global lattice (required)\n");
  printf("    -beta B                        gauge coupling (required)\n");
  printf(
      "    -ntraj N                       number of trajectories (required)\n");
  printf("    -xi0 X                         bare anisotropy (default 1.0)\n");
  printf("    -nhb N                         heatbath sweeps per trajectory "
         "(default 1, must be > 0)\n");
  printf("    -novr N                        overrelaxation sweeps per "
         "trajectory (default 4, may be 0)\n");
  printf("    -nsave N                       save gauge config every N "
         "trajectories (default 100)\n");
  printf("    -seed N                        RNG seed (default 1234)\n");
  printf("    -start FILE                    restart from a full SU(N) config\n");
  printf("    -traj-offset N                 add N to output trajectory labels\n");
  printf("    -warmup N                      save only global trajectories > N\n");
  printf("  A checkpoint is always written after the final trajectory.\n");
  printf("  NDIMS = %d.  -h, --help\n", NDIMS);
  printf("\nExample (serial): %s -latt L1 ... L_%d -beta 6.0 -ntraj 100 "
         "-nhb 1 -novr 4\n",
         prog_name, NDIMS);
  printf("Example (MPI): mpirun -np P %s -geom n_1 ... n_%d -latt L_1 ... L_%d"
         " -beta 6.0 -ntraj 100 -nhb 1 -novr 4\n",
         prog_name, NDIMS, NDIMS);
  printf("  (∏ n_i = P, each L_i divisible by n_i)\n");
}

namespace {

/// SU(3) single-process uses SOA12; MPI / other Nc use full SOA.
ArrayType defaultGaugeArrayType() {
#if NCOLORS == 3
  if (mpiCommSize() == 1) {
    return ArrayType::SOA12;
  }
#endif
  return ArrayType::SOA;
}

bool parseHeatbathCli(int argc, char **argv, int proc_grid[NDIMS],
                        std::vector<int> &lattice_size, double &beta,
                        int &ntraj, double &xi0, int &nhb, int &novr,
                        int &nsave, int &seed, int &traj_offset,
                        int &num_warmup, std::string &start_config,
                        std::string &err) {
  for (int d = 0; d < NDIMS; ++d) {
    proc_grid[d] = 1;
  }
  xi0 = 1.0;
  nhb = 1;
  novr = 4;
  nsave = 100;
  seed = 1234;
  traj_offset = 0;
  num_warmup = 0;
  start_config.clear();
  lattice_size.assign(NDIMS, 0);
  bool have_geom = false, have_latt = false, have_beta = false;
  bool have_ntraj = false, have_xi0 = false;
  bool have_nhb = false, have_novr = false, have_nsave = false;
  bool have_seed = false, have_traj_offset = false, have_warmup = false;
  bool have_start = false;

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
    if (std::strcmp(a, "-nhb") == 0) {
      if (have_nhb) {
        err = "duplicate -nhb";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-nhb requires a value";
        return false;
      }
      nhb = std::atoi(argv[i + 1]);
      have_nhb = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-novr") == 0) {
      if (have_novr) {
        err = "duplicate -novr";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-novr requires a value";
        return false;
      }
      novr = std::atoi(argv[i + 1]);
      have_novr = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-nsave") == 0) {
      if (have_nsave) {
        err = "duplicate -nsave";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-nsave requires a value";
        return false;
      }
      nsave = std::atoi(argv[i + 1]);
      have_nsave = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-seed") == 0) {
      if (have_seed) {
        err = "duplicate -seed";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-seed requires a value";
        return false;
      }
      seed = std::atoi(argv[i + 1]);
      have_seed = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-traj-offset") == 0) {
      if (have_traj_offset) {
        err = "duplicate -traj-offset";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-traj-offset requires a value";
        return false;
      }
      traj_offset = std::atoi(argv[i + 1]);
      have_traj_offset = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-warmup") == 0) {
      if (have_warmup) {
        err = "duplicate -warmup";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-warmup requires a value";
        return false;
      }
      num_warmup = std::atoi(argv[i + 1]);
      have_warmup = true;
      i += 2;
      continue;
    }
    if (std::strcmp(a, "-start") == 0) {
      if (have_start) {
        err = "duplicate -start";
        return false;
      }
      if (i + 1 >= argc) {
        err = "-start requires a value";
        return false;
      }
      start_config = argv[i + 1];
      have_start = true;
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

template <typename Real>
void runHeatbath(int ntraj, int nhb, int novr, int nsave, unsigned int seed,
                 int traj_offset, int num_warmup,
                 const std::string &start_config) {
  auto &params = PARAMS::params;
  const ArrayType array_type = defaultGaugeArrayType();

  GaugeArray<Real> gauge(array_type, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  if (mpiCommRank() == 0) {
    gauge.details();
  }

  seed += static_cast<unsigned int>(mpiCommRank());
  RandomGenerator rng(seed, params.half_volume);
  if (mpiCommRank() == 0) {
    printf("RNG initialized with seed %u\n", seed);
  }
  gauge.initCold();
  if (!start_config.empty()) {
    if (mpiCommRank() == 0) {
      printf("Restarting gauge field from %s\n", start_config.c_str());
    }
    loadGaugeBinary<Real, Real>(gauge, start_config);
  } else if (mpiCommRank() == 0) {
    printf("Initializing gauge field (cold start)...\n");
  }

  HeatBath<Real> heatbath(gauge, rng, params);
  Overrelaxation<Real> overrelax(gauge, params);
  Plaquette<Real> plaquette(gauge, params);
  PolyakovLoop<Real> polyakov(gauge, params);
  Reunitarize<Real> reunitarize(gauge, params);

  plaquette.run();
  polyakov.run();
  if (mpiCommRank() == 0) {
    printf("Initial configuration:\n");
    plaquette.printValue();
    polyakov.printValue();
    printf("\n");
  }

  std::ostringstream prefix_stream;
  prefix_stream << "su" << NCOLORS << "_nd" << NDIMS << "beta" << params.beta;
  for (int i = 0; i < NDIMS; ++i) {
    prefix_stream << "_L" << params.global_grid[i];
  }
  std::string save_prefix = prefix_stream.str();

  Timer total_timer;
  total_timer.start();

  for (int traj = 1; traj <= ntraj; ++traj) {
    const int global_traj = traj_offset + traj;
    if (mpiCommRank() == 0) {
      printf("========== Trajectory %d ==========\n", global_traj);
    }
    Timer traj_timer;
    traj_timer.start();

    // One trajectory: nhb heatbath + novr overrelax (match CUDA traj timer:
    // reunitarize is outside the measured interval).
    for (int i = 0; i < nhb; ++i) {
      heatbath.run();
    }
    for (int i = 0; i < novr; ++i) {
      overrelax.run();
    }

    traj_timer.stop();

    reunitarize.run();

    plaquette.run();
    polyakov.run();

    plaquette.printValue();
    polyakov.printValue();
    if (mpiCommRank() == 0) {
      printf("\nPerformance statistics (last sweep of each update type):\n");
    }
    if (nhb > 0) {
      heatbath.stat();
    }
    if (novr > 0) {
      overrelax.stat();
    }
    reunitarize.stat();
    plaquette.stat();
    polyakov.stat();
    if (mpiCommRank() == 0) {
      printf("Trajectory time: %.4f s\n\n", traj_timer.elapsed());
    }

    if (global_traj > num_warmup && global_traj % nsave == 0) {
      std::string filename =
          save_prefix + "_cfg_" + std::to_string(global_traj) + ".bin";
      // File format is always full SU(N) matrices (SOA12 reconstructed on
      // save).
      saveGaugeBinary<Real, Real>(gauge, filename);
    }
  }

  const int final_traj = traj_offset + ntraj;
  saveGaugeBinary<Real, Real>(
      gauge, save_prefix + "_checkpoint_" + std::to_string(final_traj) + ".bin");

  total_timer.stop();
  if (mpiCommRank() == 0) {
    printf("====================================\n");
    printf("Total simulation time: %.4f s\n", total_timer.elapsed());
    printf("====================================\n");
  }
}

int main(int argc, char *argv[]) {
  kwqft::initialize(argc, argv);

  for (int k = 1; k < argc; ++k) {
    if (std::strcmp(argv[k], "-h") == 0 ||
        std::strcmp(argv[k], "--help") == 0) {
      printUsage(argv[0]);
      kwqft::finalize();
      return 0;
    }
  }

  int proc_grid[NDIMS];
  std::vector<int> lattice_size;
  double beta = 0.0;
  int ntraj = 0;
  double xi0 = 1.0;
  int nhb = 1;
  int novr = 4;
  int nsave = 100;
  int seed = 1234;
  int traj_offset = 0;
  int num_warmup = 0;
  std::string start_config;
  std::string cli_err;
  if (!parseHeatbathCli(argc, argv, proc_grid, lattice_size, beta, ntraj, xi0,
                        nhb, novr, nsave, seed, traj_offset, num_warmup,
                        start_config, cli_err)) {
    if (mpiCommRank() == 0) {
      fprintf(stderr, "Error: %s\n", cli_err.c_str());
      printUsage(argv[0]);
    }
    kwqft::finalize();
    return 1;
  }

  if (beta <= 0.0 || ntraj <= 0 || xi0 <= 0.0) {
    fprintf(stderr, "Error: invalid beta, ntraj, or xi0\n");
    kwqft::finalize();
    return 1;
  }
  if (nhb <= 0 || novr < 0) {
    fprintf(stderr, "Error: need nhb > 0 and novr >= 0\n");
    kwqft::finalize();
    return 1;
  }
  if (nsave <= 0 || seed < 0 || traj_offset < 0 || num_warmup < 0) {
    fprintf(stderr, "Error: need nsave > 0, seed >= 0, traj-offset >= 0, and "
                    "warmup >= 0\n");
    kwqft::finalize();
    return 1;
  }

#ifdef KWQFT_USE_MPI
  int global_lattice[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_lattice[d] = lattice_size[d];
  }
  if (mpiCommSize() > 1) {
    mpiSetupCartesian(proc_grid, global_lattice);
    std::vector<int> pg(proc_grid, proc_grid + NDIMS);
    initializeParamsDistributed(lattice_size, pg, beta, true, xi0);
  } else {
    initializeParams(lattice_size, beta, true, xi0);
  }
#else
  initializeParams(lattice_size, beta, true, xi0);
#endif

  if (mpiCommRank() == 0) {
    printf("Starting SU(%d) heatbath simulation\n", NCOLORS);
    printf("Precision: %s\n",
           std::is_same_v<KWQFT_REAL_TYPE, float> ? "single" : "double");
    printf("Beta: %f\n", beta);
    printf("Number of trajectories: %d\n", ntraj);
    printf("Heatbath sweeps per trajectory (-nhb): %d\n", nhb);
    printf("Overrelaxation sweeps per trajectory (-novr): %d\n", novr);
    printf("Save interval (-nsave): %d\n", nsave);
    printf("Seed (-seed): %d\n", seed);
    printf("Trajectory offset (-traj-offset): %d\n", traj_offset);
    printf("Global warmup (-warmup): %d\n", num_warmup);
    printf("Start configuration (-start): %s\n",
           start_config.empty() ? "cold start" : start_config.c_str());
    printf("Xi0 (bare anisotropy): %f\n", xi0);
    printf("\n");
  }

  runHeatbath<KWQFT_REAL_TYPE>(ntraj, nhb, novr, nsave,
                               static_cast<unsigned int>(seed), traj_offset,
                               num_warmup, start_config);

  kwqft::finalize();
  return 0;
}
