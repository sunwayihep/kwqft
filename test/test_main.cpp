/**
 * @file test_main.cpp
 * @brief Test program for KWQFT
 *
 * Simple tests for verifying correct functionality
 */

#include "io_gauge.hpp"
#include "kwqft.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>

using namespace kwqft;

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
  if (!PARAMS::initialized) {
    initializeParams(lattice_size, 6.0, false);
  }
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
  initializeParams(lattice_size, 6.0, false);
  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, false);

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
  initializeParams(lattice_size, beta, false);
  auto &params = PARAMS::params;

  GaugeArray<Real> gauge(ArrayType::SOA, MemoryLocation::Device,
                         params.volume * NDIMS, false);
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

int main(int argc, char *argv[]) {
  kwqft::initialize(argc, argv);

  int passed = 0;
  int failed = 0;

#ifdef KWQFT_USE_MPI
  if (mpi_comm_size() > 1 && mpi_comm_rank() != 0) {
    kwqft::finalize();
    return 0;
  }
#endif

  printf("===========================================\n");
  printf("KWQFT Test Suite\n");
  printf("NCOLORS = %d, NDIMS = %d\n", NCOLORS, NDIMS);
  printf("===========================================\n\n");

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

  printf("\n===========================================\n");
  printf("Results: %d passed, %d failed\n", passed, failed);
  printf("===========================================\n");

  kwqft::finalize();
  return failed > 0 ? 1 : 0;
}
