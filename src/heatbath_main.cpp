/**
 * @file heatbath_main.cpp
 * @brief Main program for generating SU(N) gauge configurations using heatbath
 *
 * This is a Kokkos-portable version that can run on CPU or GPU
 * depending on the build configuration
 */

#include "kwqft.hpp"
#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace kwqft;

/**
 * @brief Parse command line arguments
 */
void print_usage(const char *prog_name) {
  printf("Usage: %s L1 L2 ... Ln beta ntraj\n", prog_name);
  printf("  L1, L2, ..., Ln: lattice dimensions (n = NDIMS = %d)\n", NDIMS);
  printf("  beta: gauge coupling\n");
  printf("  ntraj: number of trajectories\n");
  printf("\nExample: %s 8 8 8 16 6.0 100\n", prog_name);
}

template <typename Real>
void run_heatbath(const std::vector<int> &lattice_size, double beta,
                  int ntraj) {

  // Initialize parameters
  initializeParams(lattice_size, beta, true);
  auto &params = PARAMS::params;

  // Create gauge field
  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  gauge.details();

  // Initialize random number generator
  unsigned int seed = 1234;
  RandomGenerator rng(seed, params.half_volume);
  printf("RNG initialized with seed %u\n", seed);

  // Cold start
  printf("Initializing gauge field (cold start)...\n");
  gauge.initCold();

  // Create update and measurement objects
  HeatBath<Real> heatbath(gauge, rng, params);
  Reunitarize<Real> reunitarize(gauge, params);
  Plaquette<Real> plaquette(gauge, params);
  PolyakovLoop<Real> polyakov(gauge, params);

  // Initial measurement
  plaquette.run();
  polyakov.run();
  printf("Initial configuration:\n");
  plaquette.printValue();
  polyakov.printValue();
  printf("\n");

  // Configuration save settings
  int num_warmup = 0;
  int save_interval = 100;
  std::ostringstream prefix_stream;
  prefix_stream << "su" << NCOLORS << "_nd" << NDIMS << "_beta" << beta;
  for (int i = 0; i < NDIMS; ++i) {
    prefix_stream << "_L" << lattice_size[i];
  }
  std::string save_prefix = prefix_stream.str();

  // Main trajectory loop
  Timer total_timer;
  total_timer.start();

  for (int traj = 1; traj <= ntraj; ++traj) {
    printf("========== Trajectory %d ==========\n", traj);
    Timer traj_timer;
    traj_timer.start();

    // Heatbath update
    heatbath.run();

    // Reunitarize
    reunitarize.run();

    traj_timer.stop();

    // Measurements
    plaquette.run();
    polyakov.run();

    plaquette.printValue();
    polyakov.printValue();

    printf("\nPerformance statistics:\n");
    heatbath.stat();
    reunitarize.stat();
    plaquette.stat();
    polyakov.stat();
    printf("Trajectory time: %.4f s\n\n", traj_timer.elapsed());

    // Save configuration
    if (traj > num_warmup && traj % save_interval == 0) {
      std::string filename =
          save_prefix + "_cfg_" + std::to_string(traj) + ".bin";
      // Note: save_gauge_binary is declared in io_gauge.cpp
      // For now, just print a message
      printf("Would save configuration to: %s\n", filename.c_str());
    }
  }

  total_timer.stop();
  printf("====================================\n");
  printf("Total simulation time: %.4f s\n", total_timer.elapsed());
  printf("====================================\n");

  // Cleanup is automatic via RAII
}

int main(int argc, char *argv[]) {
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    // Initialize KWQFT
    kwqft::initialize(argc, argv);

    // Parse command line arguments
    if (argc != NDIMS + 3) {
      printf("Error: Expected %d arguments, got %d\n", NDIMS + 2, argc - 1);
      print_usage(argv[0]);
      Kokkos::finalize();
      return 1;
    }

    // Parse lattice dimensions
    std::vector<int> lattice_size(NDIMS);
    for (int i = 0; i < NDIMS; ++i) {
      lattice_size[i] = atoi(argv[i + 1]);
      if (lattice_size[i] <= 0) {
        printf("Error: Invalid lattice dimension L%d = %d\n", i + 1,
               lattice_size[i]);
        Kokkos::finalize();
        return 1;
      }
    }

    // Parse beta and ntraj
    double beta = atof(argv[NDIMS + 1]);
    int ntraj = atoi(argv[NDIMS + 2]);

    if (beta <= 0) {
      printf("Error: Invalid beta = %f\n", beta);
      Kokkos::finalize();
      return 1;
    }

    if (ntraj <= 0) {
      printf("Error: Invalid ntraj = %d\n", ntraj);
      Kokkos::finalize();
      return 1;
    }

    // Print configuration
    printf("Starting SU(%d) heatbath simulation\n", NCOLORS);
    printf("Lattice: ");
    for (int i = 0; i < NDIMS; ++i) {
      printf("%d", lattice_size[i]);
      if (i < NDIMS - 1)
        printf(" x ");
    }
    printf("\n");
    printf("Beta: %f\n", beta);
    printf("Number of trajectories: %d\n", ntraj);
    printf("\n");

    // Run simulation with double precision
    run_heatbath<double>(lattice_size, beta, ntraj);

    kwqft::finalize();
  }

  Kokkos::finalize();
  return 0;
}
