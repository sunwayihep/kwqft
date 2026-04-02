/**
 * @file heatbath_main.cpp
 * @brief Main program for generating SU(N) gauge configurations using heatbath
 *
 * This is a Kokkos-portable version that can run on CPU or GPU
 * depending on the build configuration
 *
 * MPI (optional): \c mpirun -np P ./heatbath -geom p0 p1 ... p_{n-1} L0 ...
 * L_{n-1} beta ntraj [xi0] with ∏ p_i = P and L_i divisible by p_i
 * (Chroma-style).
 */

#include "kwqft.hpp"
#ifdef KWQFT_USE_MPI
#include "gauge_halo.hpp"
#endif
#include "mpi_layout.hpp"

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace kwqft;

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s [-geom|--geom p0 p1 ... p_{n-1}] L0 L1 ... L_{n-1} beta ntraj "
         "[xi0]\n",
         prog_name);
  printf("  NDIMS = %d. Optional -geom: MPI process grid (MPI builds only);\n",
         NDIMS);
  printf("  then global lattice sizes L0..L_{n-1}, beta, trajectory count, "
         "optional xi0.\n");
  printf("\nExample (serial): %s 8 8 8 16 6.0 100\n", prog_name);
  printf(
      "Example (MPI, 8 ranks): mpirun -np 8 %s -geom 1 2 2 2 4 4 4 8 6.0 10\n",
      prog_name);
}

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
  int save_interval = 100;
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

    if (traj > num_warmup && traj % save_interval == 0 &&
        mpi_comm_rank() == 0) {
      std::string filename =
          save_prefix + "_cfg_" + std::to_string(traj) + ".bin";
      printf("Would save configuration to: %s\n", filename.c_str());
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

  int proc_grid[NDIMS];
  std::vector<std::string> pos;
  if (!parse_geom_argv(argc, argv, proc_grid, pos)) {
    fprintf(stderr, "Error: invalid -geom (need %d integers after -geom)\n",
            NDIMS);
    print_usage(argv[0]);
    kwqft::finalize();
    return 1;
  }

  const int npos = static_cast<int>(pos.size());
  if (npos != NDIMS + 2 && npos != NDIMS + 3) {
    fprintf(stderr, "Error: expected %d or %d positional args, got %d\n",
            NDIMS + 2, NDIMS + 3, npos);
    print_usage(argv[0]);
    kwqft::finalize();
    return 1;
  }

  std::vector<int> lattice_size(NDIMS);
  for (int i = 0; i < NDIMS; ++i) {
    lattice_size[i] = std::atoi(pos[static_cast<size_t>(i)].c_str());
    if (lattice_size[i] <= 0) {
      fprintf(stderr, "Error: invalid lattice dimension L%d\n", i);
      kwqft::finalize();
      return 1;
    }
  }

  double beta = std::atof(pos[static_cast<size_t>(NDIMS)].c_str());
  int ntraj = std::atoi(pos[static_cast<size_t>(NDIMS + 1)].c_str());
  double xi0 = 1.0;
  if (npos == NDIMS + 3) {
    xi0 = std::atof(pos[static_cast<size_t>(NDIMS + 2)].c_str());
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
