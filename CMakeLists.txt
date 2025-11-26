cmake_minimum_required(VERSION 3.16)
project(KWQFT CXX)

# KWQFT - Kokkos Ken Wilson Quantum Field Theory
# Lattice gauge theory library with performance portability

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(KWQFT_ENABLE_OPENMP "Enable OpenMP backend" OFF)
option(KWQFT_ENABLE_CUDA "Enable CUDA backend" OFF)
option(KWQFT_ENABLE_HIP "Enable HIP backend" OFF)
option(KWQFT_ENABLE_SYCL "Enable SYCL backend" OFF)

# Lattice parameters (can be overridden at compile time)
set(KWQFT_NCOLORS "3" CACHE STRING "Number of colors for SU(N)")
set(KWQFT_NDIMS "4" CACHE STRING "Number of spacetime dimensions")

# Local Kokkos source directory (use this if you cannot access GitHub)
set(KOKKOS_SOURCE_DIR "" CACHE PATH "Path to local Kokkos source directory")
set(KOKKOS_KERNELS_SOURCE_DIR "" CACHE PATH "Path to local KokkosKernels source directory (optional)")

# Kokkos configuration - must be set before adding Kokkos
if(KWQFT_ENABLE_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "")
    # Set architecture - can be overridden via -DKokkos_ARCH_XXX=ON
    if(NOT DEFINED Kokkos_ARCH_AMPERE80 AND NOT DEFINED Kokkos_ARCH_VOLTA70 AND 
       NOT DEFINED Kokkos_ARCH_TURING75 AND NOT DEFINED Kokkos_ARCH_HOPPER90)
        set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "Default GPU architecture")
    endif()
endif()

if(KWQFT_ENABLE_HIP)
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "")
endif()

if(KWQFT_ENABLE_SYCL)
    set(Kokkos_ENABLE_SYCL ON CACHE BOOL "")
endif()

if(KWQFT_ENABLE_OPENMP)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
else()
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
endif()

#=============================================================================
# Find or build Kokkos
#=============================================================================
find_package(Kokkos QUIET)

if(Kokkos_FOUND)
    message(STATUS "Found installed Kokkos: ${Kokkos_DIR}")
elseif(KOKKOS_SOURCE_DIR AND EXISTS "${KOKKOS_SOURCE_DIR}/CMakeLists.txt")
    # Use local Kokkos source directory
    message(STATUS "Using local Kokkos source: ${KOKKOS_SOURCE_DIR}")
    add_subdirectory(${KOKKOS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/kokkos)
else()
    # Try to fetch from GitHub (will fail if no network access)
    message(STATUS "Kokkos not found and no local source specified")
    message(STATUS "Attempting to fetch Kokkos from GitHub...")
    message(STATUS "If this fails, please specify -DKOKKOS_SOURCE_DIR=/path/to/kokkos")
    include(FetchContent)
    FetchContent_Declare(
        kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG 4.7.01
    )
    FetchContent_MakeAvailable(kokkos)
endif()

#=============================================================================
# Find or build KokkosKernels (optional)
#=============================================================================
find_package(KokkosKernels QUIET)

if(KokkosKernels_FOUND)
    message(STATUS "Found installed KokkosKernels: ${KokkosKernels_DIR}")
    set(HAVE_KOKKOS_KERNELS TRUE)
elseif(KOKKOS_KERNELS_SOURCE_DIR AND EXISTS "${KOKKOS_KERNELS_SOURCE_DIR}/CMakeLists.txt")
    # Use local KokkosKernels source directory
    message(STATUS "Using local KokkosKernels source: ${KOKKOS_KERNELS_SOURCE_DIR}")
    add_subdirectory(${KOKKOS_KERNELS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/kokkos-kernels)
    set(HAVE_KOKKOS_KERNELS TRUE)
else()
    # KokkosKernels is optional - Kokkos itself has random number support
    message(STATUS "KokkosKernels not found (optional, Kokkos has built-in RNG support)")
    set(HAVE_KOKKOS_KERNELS FALSE)
endif()

# Add compile definitions
add_compile_definitions(NCOLORS=${KWQFT_NCOLORS})
add_compile_definitions(NDIMS=${KWQFT_NDIMS})

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(KWQFT_SOURCES
    src/constants.cpp
    src/gauge_array.cpp
    src/random.cpp
    src/monte.cpp
    src/plaquette.cpp
    src/polyakov.cpp
    src/reunitarize.cpp
    src/io_gauge.cpp
)

# Create library
add_library(kwqft ${KWQFT_SOURCES})
target_link_libraries(kwqft PUBLIC Kokkos::kokkos)

# Link KokkosKernels if available
if(HAVE_KOKKOS_KERNELS AND TARGET Kokkos::kokkoskernels)
    target_link_libraries(kwqft PUBLIC Kokkos::kokkoskernels)
    target_compile_definitions(kwqft PUBLIC KWQFT_USE_KOKKOS_KERNELS)
    message(STATUS "Linking with KokkosKernels")
endif()

# Main executable
add_executable(heatbath src/heatbath_main.cpp)
target_link_libraries(heatbath kwqft)

# Test executable
add_executable(test_kwqft test/test_main.cpp)
target_link_libraries(test_kwqft kwqft)

# Installation
install(TARGETS kwqft heatbath
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include/kwqft)

# Print configuration
message(STATUS "===========================================")
message(STATUS "KWQFT Configuration:")
message(STATUS "  NCOLORS: ${KWQFT_NCOLORS}")
message(STATUS "  NDIMS: ${KWQFT_NDIMS}")
message(STATUS "  OpenMP: ${KWQFT_ENABLE_OPENMP}")
message(STATUS "  CUDA: ${KWQFT_ENABLE_CUDA}")
message(STATUS "  HIP: ${KWQFT_ENABLE_HIP}")
message(STATUS "  SYCL: ${KWQFT_ENABLE_SYCL}")
if(KOKKOS_SOURCE_DIR)
    message(STATUS "  Kokkos source: ${KOKKOS_SOURCE_DIR}")
endif()
if(KOKKOS_KERNELS_SOURCE_DIR)
    message(STATUS "  KokkosKernels source: ${KOKKOS_KERNELS_SOURCE_DIR}")
endif()
message(STATUS "===========================================")
message(STATUS "")
message(STATUS "Usage hints for offline build:")
message(STATUS "  -DKOKKOS_SOURCE_DIR=/path/to/kokkos")
message(STATUS "  -DKOKKOS_KERNELS_SOURCE_DIR=/path/to/kokkos-kernels (optional)")
message(STATUS "")

