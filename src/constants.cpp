/**
 * @file constants.cpp
 * @brief Implementation of global lattice parameters
 */

#include "constants.hpp"
#include <cstdio>
#include <memory>

namespace kwqft {

// Global host parameters (plain struct, no Kokkos dependency)
namespace PARAMS {
    LatticeParams params;
    bool initialized = false;
}

// Use unique_ptr for lazy initialization of Kokkos Views
// This avoids the static initialization order fiasco
static std::unique_ptr<ParamsView> s_device_params;
static std::unique_ptr<ParamsHostView> s_host_params_mirror;

ParamsView& get_device_params() {
    if (!s_device_params) {
        s_device_params = std::make_unique<ParamsView>("device_params");
    }
    return *s_device_params;
}

ParamsHostView& get_host_params_mirror() {
    if (!s_host_params_mirror) {
        s_host_params_mirror = std::make_unique<ParamsHostView>(
            Kokkos::create_mirror_view(get_device_params())
        );
    }
    return *s_host_params_mirror;
}

void initializeParams(const std::vector<int>& lattice_size, double beta, bool verbose) {
    if (PARAMS::initialized) {
        KWQFT_WARNING("Parameters already initialized");
        return;
    }

    PARAMS::params.initialize(lattice_size, beta);
    PARAMS::initialized = true;

    // Initialize device params view (lazy)
    auto& device_params = get_device_params();
    auto& host_mirror = get_host_params_mirror();
    
    // Copy to host mirror
    host_mirror() = PARAMS::params;
    
    // Copy to device
    Kokkos::deep_copy(device_params, host_mirror);

    if (verbose) {
        print_params();
    }
}

void copy_params_to_device() {
    if (!PARAMS::initialized) {
        KWQFT_ERROR("Parameters not initialized");
        return;
    }
    
    auto& device_params = get_device_params();
    auto& host_mirror = get_host_params_mirror();
    
    host_mirror() = PARAMS::params;
    Kokkos::deep_copy(device_params, host_mirror);
}

void print_params() {
    printf("==========================================================\n");
    printf("Lattice Parameters:\n");
    printf("  Dimensions: ");
    for (int i = 0; i < NDIMS; ++i) {
        printf("%d", PARAMS::params.grid[i]);
        if (i < NDIMS - 1) printf(" x ");
    }
    printf("\n");
    printf("  Volume: %d\n", PARAMS::params.volume);
    printf("  Beta: %.6f\n", PARAMS::params.beta);
    printf("  Beta/Nc: %.6f\n", PARAMS::params.beta_over_nc);
    printf("  Number of colors: %d\n", NCOLORS);
    printf("  Number of dimensions: %d\n", NDIMS);
    printf("==========================================================\n");
}

void finalizeParams() {
    // Release Kokkos views before Kokkos::finalize()
    s_host_params_mirror.reset();
    s_device_params.reset();
    PARAMS::initialized = false;
}

} // namespace kwqft
