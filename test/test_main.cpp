/**
 * @file test_main.cpp
 * @brief Test program for KWQFT
 *
 * Simple tests for verifying correct functionality
 */

#include "io_gauge.hpp"
#include "kwqft.hpp"
#ifdef KWQFT_USE_MPI
#include "mpi_layout.hpp"
#include <mpi.h>
#endif
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <memory>

using namespace kwqft;

void reset_and_initialize_params(const std::vector<int> &lattice_size,
                                 double beta, double xi0 = 1.0) {
  if (PARAMS::initialized) {
    finalizeParams();
  }
  initializeParams(lattice_size, beta, false, xi0);
}

/// Fill proc_grid with 1s, then set proc_grid[split_dim] = nproc_along_dim.
inline void fill_proc_grid(int proc_grid[NDIMS], int split_dim,
                           int nproc_along_dim = 2) {
  for (int d = 0; d < NDIMS; ++d) {
    proc_grid[d] = 1;
  }
  proc_grid[split_dim] = nproc_along_dim;
}

template <typename Real> bool test_complex() {
  printf("Testing Complex<Real>...\n");

  Complex<Real> a(3, 4);
  Complex<Real> b(1, 2);

  // Test addition
  Complex<Real> c = a + b;
  if (std::abs(c.real() - 4) > 1e-10 || std::abs(c.imag() - 6) > 1e-10) {
    printf("  FAILED: addition\n");
    return false;
  }

  // Test multiplication
  c = a * b;
  if (std::abs(c.real() - (-5)) > 1e-10 || std::abs(c.imag() - 10) > 1e-10) {
    printf("  FAILED: multiplication\n");
    return false;
  }

  // Test abs
  if (std::abs(a.abs() - 5) > 1e-10) {
    printf("  FAILED: abs\n");
    return false;
  }

  // Test conjugate
  c = ~a;
  if (std::abs(c.real() - 3) > 1e-10 || std::abs(c.imag() - (-4)) > 1e-10) {
    printf("  FAILED: conjugate\n");
    return false;
  }

  printf("  PASSED\n");
  return true;
}

template <typename Real> bool test_matrix() {
  printf("Testing MatrixSun<Real, %d>...\n", NCOLORS);

  using Matrix = MatrixSun<Real, NCOLORS>;

  // Test identity
  Matrix I = Matrix::identity();
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      Real expected_re = (i == j) ? Real(1) : Real(0);
      if (std::abs(I.e[i][j].real() - expected_re) > 1e-10 ||
          std::abs(I.e[i][j].imag()) > 1e-10) {
        printf("  FAILED: identity\n");
        return false;
      }
    }
  }

  // Test multiplication by identity
  Matrix A;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      A.e[i][j] = Complex<Real>(i + j, i - j);
    }
  }

  Matrix B = A * I;
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      if (std::abs(B.e[i][j].real() - A.e[i][j].real()) > 1e-10 ||
          std::abs(B.e[i][j].imag() - A.e[i][j].imag()) > 1e-10) {
        printf("  FAILED: multiplication by identity\n");
        return false;
      }
    }
  }

  // Test trace
  Complex<Real> tr = I.trace();
  if (std::abs(tr.real() - NCOLORS) > 1e-10 || std::abs(tr.imag()) > 1e-10) {
    printf("  FAILED: trace\n");
    return false;
  }

  // Test dagger
  Matrix Ad = A.dagger();
  for (int i = 0; i < NCOLORS; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      if (std::abs(Ad.e[i][j].real() - A.e[j][i].real()) > 1e-10 ||
          std::abs(Ad.e[i][j].imag() + A.e[j][i].imag()) > 1e-10) {
        printf("  FAILED: dagger\n");
        return false;
      }
    }
  }

  printf("  PASSED\n");
  return true;
}

template <typename Real> bool test_gauge_io_roundtrip() {
  printf("Testing gauge configuration I/O round-trip...\n");

  std::vector<int> lattice_size(NDIMS, 4);
  reset_and_initialize_params(lattice_size, 6.0);
  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  gauge.initCold();

  RandomGenerator rng(54321, params.half_volume);
  HeatBath<Real> heatbath(gauge, rng, params);
  Reunitarize<Real> reunitarize(gauge, params);
  Plaquette<Real> plaq(gauge, params);

  const int n_sweeps = 5;
  for (int i = 0; i < n_sweeps; ++i) {
    heatbath.run();
    reunitarize.run();
  }

  plaq.run();
  const Real plaq_before = plaq.value();

  const std::string filename = "test_io_roundtrip.bin";
  save_gauge_binary<Real, Real>(gauge, filename, false);
  load_gauge_binary<Real, Real>(gauge, filename, false);

  plaq.run();
  const Real plaq_after = plaq.value();

  const Real diff = std::abs(plaq_before - plaq_after);
  const Real tol = Real(1e-10);
  if (diff > tol) {
    printf("  FAILED: plaquette mismatch after save/load\n");
    printf("    before save   = %f\n", static_cast<double>(plaq_before));
    printf("    after reload  = %f\n", static_cast<double>(plaq_after));
    printf("    |difference|  = %e (tolerance %e)\n",
           static_cast<double>(diff), static_cast<double>(tol));
    return false;
  }

  printf("  Plaquette before save  = %f\n", static_cast<double>(plaq_before));
  printf("  Plaquette after reload = %f (|diff| = %e)\n",
         static_cast<double>(plaq_after), static_cast<double>(diff));
  printf("  PASSED\n");
  return true;
}

template <typename Real> bool test_gauge_cold_start() {
  printf("Testing GaugeArray cold start...\n");

  // Create a small lattice
  std::vector<int> lattice_size(NDIMS, 4);
  reset_and_initialize_params(lattice_size, 6.0);
  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);

  gauge.initCold();

  // Compute plaquette - should be 1.0 for cold start
  Plaquette<Real> plaq(gauge, params);
  plaq.run();

  Real plaq_value = plaq.value();
  if (std::abs(plaq_value - Real(1)) > 1e-6) {
    printf("  FAILED: expected plaquette = 1.0, got %f\n",
           static_cast<double>(plaq_value));
    return false;
  }

  printf("  Plaquette = %f (expected 1.0)\n", static_cast<double>(plaq_value));
  printf("  PASSED\n");
  return true;
}

template <typename Real> bool test_heatbath_thermalization() {
  printf("Testing HeatBath thermalization...\n");

  // Create a small lattice
  std::vector<int> lattice_size(NDIMS, 4);
  double beta = 6.0;
  reset_and_initialize_params(lattice_size, beta);
  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  gauge.initCold();

  RandomGenerator rng(12345, params.half_volume);

  HeatBath<Real> heatbath(gauge, rng, params);
  Plaquette<Real> plaq(gauge, params);

  // Run a few heatbath sweeps
  int n_sweeps = 10;
  for (int i = 0; i < n_sweeps; ++i) {
    heatbath.run();
  }

  plaq.run();
  Real plaq_value = plaq.value();

  // For beta=6.0 in 4D, the plaquette should be around 0.59-0.61
  // For a small lattice and few sweeps, allow a wider range
  if (plaq_value < 0.3 || plaq_value > 1.0) {
    printf("  FAILED: plaquette = %f is outside reasonable range\n",
           static_cast<double>(plaq_value));
    return false;
  }

  printf("  After %d sweeps: plaquette = %f\n", n_sweeps,
         static_cast<double>(plaq_value));
  printf("  PASSED (plaquette in reasonable range)\n");
  return true;
}

#ifdef KWQFT_USE_MPI
template <typename Real> bool test_mpi_gauge_io_roundtrip() {
  const int nproc = mpi_comm_size();
  const int rank = mpi_comm_rank();

  if (nproc < 2) {
    if (rank == 0) {
      printf("Testing MPI gauge configuration I/O round-trip...\n");
      printf("  SKIPPED (need >= 2 MPI ranks)\n");
    }
    return true;
  }
  if (nproc != 2) {
    if (rank == 0) {
      printf("Testing MPI gauge configuration I/O round-trip...\n");
      printf("  SKIPPED (need exactly 2 MPI ranks)\n");
    }
    return true;
  }

  if (rank == 0) {
    printf("Testing MPI gauge configuration I/O round-trip...\n");
  }

  int proc_grid[NDIMS];
  fill_proc_grid(proc_grid, 0, 2); // split along x
  std::vector<int> global_lattice(NDIMS, 4);
  int global_lattice_arr[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_lattice_arr[d] = global_lattice[d];
  }

  if (PARAMS::initialized) {
    finalizeParams();
  }
  mpi_setup_cartesian(proc_grid, global_lattice_arr);
  std::vector<int> pg(proc_grid, proc_grid + NDIMS);
  initializeParamsDistributed(global_lattice, pg, 6.0, false);

  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  gauge.initCold();

  RandomGenerator rng(424242u + static_cast<unsigned int>(rank),
                    params.half_volume);
  HeatBath<Real> heatbath(gauge, rng, params);
  Reunitarize<Real> reunitarize(gauge, params);
  Plaquette<Real> plaq(gauge, params);

  const int n_sweeps = 3;
  for (int i = 0; i < n_sweeps; ++i) {
    heatbath.run();
    reunitarize.run();
  }

  plaq.run();
  const Real plaq_before = plaq.value();

  const std::string filename = "test_mpi_io_roundtrip.bin";
  save_gauge_binary<Real, Real>(gauge, filename, false);
  load_gauge_binary<Real, Real>(gauge, filename, false);

  plaq.run();
  const Real plaq_after = plaq.value();

  const Real diff = std::abs(plaq_before - plaq_after);
  const Real tol = Real(1e-10);
  bool ok = diff <= tol;
  if (rank == 0) {
    if (!ok) {
      printf("  FAILED: plaquette mismatch after MPI save/load\n");
      printf("    before save   = %f\n", static_cast<double>(plaq_before));
      printf("    after reload  = %f\n", static_cast<double>(plaq_after));
      printf("    |difference|  = %e (tolerance %e)\n",
             static_cast<double>(diff), static_cast<double>(tol));
    } else {
      printf("  Plaquette before save  = %f\n",
             static_cast<double>(plaq_before));
      printf("  Plaquette after reload = %f (|diff| = %e)\n",
             static_cast<double>(plaq_after), static_cast<double>(diff));
      printf("  PASSED\n");
    }
  }

  int ok_int = ok ? 1 : 0;
  MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok_int != 0;
}

template <typename Real> bool test_mpi_polyakov_time_split() {
  const int nproc = mpi_comm_size();
  const int rank = mpi_comm_rank();

  if (nproc < 2) {
    if (rank == 0) {
      printf("Testing MPI Polyakov loop (time direction split)...\n");
      printf("  SKIPPED (need >= 2 MPI ranks)\n");
    }
    return true;
  }
  if (nproc != 2) {
    if (rank == 0) {
      printf("Testing MPI Polyakov loop (time direction split)...\n");
      printf("  SKIPPED (need exactly 2 MPI ranks)\n");
    }
    return true;
  }

  if (rank == 0) {
    printf("Testing MPI Polyakov loop (time direction split)...\n");
  }

  int proc_grid[NDIMS];
  fill_proc_grid(proc_grid, NDIMS - 1, 2); // split along time
  std::vector<int> global_lattice(NDIMS, 4);
  global_lattice[NDIMS - 1] = 8;
  int global_lattice_arr[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_lattice_arr[d] = global_lattice[d];
  }

  if (PARAMS::initialized) {
    finalizeParams();
  }
  mpi_setup_cartesian(proc_grid, global_lattice_arr);
  std::vector<int> pg(proc_grid, proc_grid + NDIMS);
  initializeParamsDistributed(global_lattice, pg, 6.0, false);

  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  gauge.initCold();

  PolyakovLoop<Real> polyakov(gauge, params);
  polyakov.run();

  const Real cold_abs = polyakov.absValue();
  const Real cold_tol = Real(1e-6);
  bool ok = std::abs(cold_abs - Real(1)) <= cold_tol;

  if (ok) {
    RandomGenerator rng(777u + static_cast<unsigned int>(rank), params.half_volume);
    HeatBath<Real> heatbath(gauge, rng, params);
    for (int i = 0; i < 5; ++i) {
      heatbath.run();
    }
    polyakov.run();
    const Real hot_abs = polyakov.absValue();
    ok = hot_abs > Real(0) && hot_abs < Real(1.01);
    if (rank == 0) {
      if (!ok) {
        printf("  FAILED: thermalized |P| = %f out of range\n",
               static_cast<double>(hot_abs));
      } else {
        printf("  Cold start |P| = %f (expected 1.0)\n",
               static_cast<double>(cold_abs));
        printf("  After 5 sweeps |P| = %f\n", static_cast<double>(hot_abs));
        printf("  PASSED\n");
      }
    }
  } else if (rank == 0) {
    printf("  FAILED: cold start |P| = %f (expected 1.0)\n",
           static_cast<double>(cold_abs));
  }

  int ok_int = ok ? 1 : 0;
  MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok_int != 0;
}

template <typename Real> bool test_mpi_shift_heatbath() {
  const int nproc = mpi_comm_size();
  const int rank = mpi_comm_rank();

  if (nproc < 2) {
    if (rank == 0) {
      printf("Testing MPI shift-based HeatBath...\n");
      printf("  SKIPPED (need >= 2 MPI ranks)\n");
    }
    return true;
  }
  if (nproc != 2) {
    if (rank == 0) {
      printf("Testing MPI shift-based HeatBath...\n");
      printf("  SKIPPED (need exactly 2 MPI ranks)\n");
    }
    return true;
  }

  if (rank == 0) {
    printf("Testing MPI shift-based HeatBath...\n");
  }

  int proc_grid[NDIMS];
  fill_proc_grid(proc_grid, 0, 2); // split along x
  std::vector<int> global_lattice(NDIMS, 8);
  int global_lattice_arr[NDIMS];
  for (int d = 0; d < NDIMS; ++d) {
    global_lattice_arr[d] = global_lattice[d];
  }

  if (PARAMS::initialized) {
    finalizeParams();
  }
  mpi_setup_cartesian(proc_grid, global_lattice_arr);
  std::vector<int> pg(proc_grid, proc_grid + NDIMS);
  initializeParamsDistributed(global_lattice, pg, 6.0, false);

  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, true);
  gauge.initCold();

  Plaquette<Real> plaq(gauge, params);
  plaq.run();
  const Real cold_plaq = plaq.value();

  RandomGenerator rng(13579u + static_cast<unsigned int>(rank), params.half_volume);
  HeatBath<Real> heatbath(gauge, rng, params);
  for (int i = 0; i < 3; ++i) {
    heatbath.run();
  }
  plaq.run();
  const Real hot_plaq = plaq.value();

  bool ok = std::abs(cold_plaq - Real(1)) <= Real(1e-6);
  ok = ok && hot_plaq > Real(0.4) && hot_plaq < Real(0.95);

  if (rank == 0) {
    if (!ok) {
      printf("  FAILED: cold plaquette = %f, after 3 sweeps = %f\n",
             static_cast<double>(cold_plaq), static_cast<double>(hot_plaq));
    } else {
      printf("  Cold plaquette = %f (expected 1.0)\n",
             static_cast<double>(cold_plaq));
      printf("  After 3 shift-based sweeps plaquette = %f\n",
             static_cast<double>(hot_plaq));
      printf("  PASSED\n");
    }
  }

  int ok_int = ok ? 1 : 0;
  MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok_int != 0;
}
#endif

int main(int argc, char *argv[]) {
  kwqft::initialize(argc, argv);

  int passed = 0;
  int failed = 0;
  const bool is_primary = (mpi_comm_rank() == 0);

#ifdef KWQFT_USE_MPI
  const bool run_serial_suite = (mpi_comm_size() == 1) && is_primary;
#else
  const bool run_serial_suite = true;
#endif

  if (is_primary) {
    printf("===========================================\n");
    printf("KWQFT Test Suite\n");
    printf("NCOLORS = %d, NDIMS = %d\n", NCOLORS, NDIMS);
    printf("===========================================\n\n");
  }

  if (run_serial_suite) {
    if (test_complex<double>())
      passed++;
    else
      failed++;
    if (test_matrix<double>())
      passed++;
    else
      failed++;
    if (test_gauge_io_roundtrip<double>())
      passed++;
    else
      failed++;
    if (test_gauge_cold_start<double>())
      passed++;
    else
      failed++;
    if (test_heatbath_thermalization<double>())
      passed++;
    else
      failed++;
  }

#ifdef KWQFT_USE_MPI
  if (mpi_comm_size() > 1) {
    if (test_mpi_gauge_io_roundtrip<double>())
      passed++;
    else
      failed++;
    if (test_mpi_polyakov_time_split<double>())
      passed++;
    else
      failed++;
    if (test_mpi_shift_heatbath<double>())
      passed++;
    else
      failed++;
  }
#endif

  if (is_primary) {
    printf("\n===========================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    printf("===========================================\n");
  }

  int exit_code = failed > 0 ? 1 : 0;
#ifdef KWQFT_USE_MPI
  if (mpi_comm_size() > 1) {
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
#endif

  kwqft::finalize();
  return exit_code;
}
