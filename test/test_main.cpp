/**
 * @file test_main.cpp
 * @brief Test program for KWQFT
 * 
 * Simple tests for verifying correct functionality
 */

#include <Kokkos_Core.hpp>
#include "kwqft.hpp"
#include <cstdio>
#include <cmath>

using namespace kwqft;

template<typename Real>
bool test_complex() {
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

template<typename Real>
bool test_matrix() {
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

template<typename Real>
bool test_gauge_cold_start() {
    printf("Testing GaugeArray cold start...\n");
    
    // Create a small lattice
    std::vector<int> lattice_size(NDIMS, 4);
    initializeParams(lattice_size, 6.0, false);
    auto& params = PARAMS::params;
    
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

template<typename Real>
bool test_heatbath_thermalization() {
    printf("Testing HeatBath thermalization...\n");
    
    // Create a small lattice
    std::vector<int> lattice_size(NDIMS, 4);
    double beta = 6.0;
    initializeParams(lattice_size, beta, false);
    auto& params = PARAMS::params;
    
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

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    
    {
        printf("===========================================\n");
        printf("KWQFT Test Suite\n");
        printf("NCOLORS = %d, NDIMS = %d\n", NCOLORS, NDIMS);
        printf("===========================================\n\n");
        
        int passed = 0;
        int failed = 0;
        
        // Run tests
        if (test_complex<double>()) passed++; else failed++;
        if (test_matrix<double>()) passed++; else failed++;
        if (test_gauge_cold_start<double>()) passed++; else failed++;
        if (test_heatbath_thermalization<double>()) passed++; else failed++;
        
        printf("\n===========================================\n");
        printf("Results: %d passed, %d failed\n", passed, failed);
        printf("===========================================\n");
        
        // Cleanup before Kokkos::finalize()
        finalizeParams();
        
        if (failed > 0) {
            Kokkos::finalize();
            return 1;
        }
    }
    
    Kokkos::finalize();
    return 0;
}

