/**
 * @file kwqft_common.hpp
 * @brief Common definitions and utilities for KWQFT
 *
 * KWQFT - Kokkos-based Ken Wilson Quantum Field Theory on lattice
 *
 * This file provides portable definitions that work across
 * different Kokkos backends (CUDA, HIP, OpenMP, Serial, SYCL)
 */

#ifndef KWQFT_COMMON_HPP
#define KWQFT_COMMON_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cmath>
#include <cstdio>

namespace kwqft {

//=============================================================================
// Configuration macros
//=============================================================================

#ifndef NCOLORS
#warning "NCOLORS not defined, using default value 3"
#define NCOLORS 3
#endif

#ifndef NDIMS
#warning "NDIMS not defined, using default value 4"
#define NDIMS 4
#endif

// Total number of SU(2) subgroups
#define TOTAL_SUB_BLOCKS ((NCOLORS) * ((NCOLORS) - 1) / 2)

// Plaquette counts
#define TOTAL_NUM_PLAQS ((NDIMS * (NDIMS - 1)) / 2)
#define TOTAL_NUM_TPLAQS (NDIMS - 1)
#define TOTAL_NUM_SPLAQS (TOTAL_NUM_PLAQS - TOTAL_NUM_TPLAQS)

//=============================================================================
// Mathematical constants
//=============================================================================

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef PII
#define PII 6.2831853071795864769252867665590
#endif

//=============================================================================
// Kokkos type aliases for execution and memory spaces
//=============================================================================

// Default execution space (will be CUDA, HIP, OpenMP, or Serial)
using DefaultExecSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemSpace = typename DefaultExecSpace::memory_space;

// Host execution and memory space
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemSpace = typename HostExecSpace::memory_space;

// Scratch memory space for team-level scratch
using ScratchMemSpace = typename DefaultExecSpace::scratch_memory_space;

// Range policy for parallel loops
using range_policy = Kokkos::RangePolicy<DefaultExecSpace>;
using host_range_policy = Kokkos::RangePolicy<HostExecSpace>;

// Team policy for hierarchical parallelism
using team_policy = Kokkos::TeamPolicy<DefaultExecSpace>;
using team_member = typename team_policy::member_type;

// MDRange policy for multi-dimensional parallel loops
template <int Rank>
using md_range_policy =
    Kokkos::MDRangePolicy<DefaultExecSpace, Kokkos::Rank<Rank>>;

//=============================================================================
// View type aliases
//=============================================================================

// 1D Views
template <typename T> using View1D = Kokkos::View<T *, DefaultMemSpace>;

template <typename T> using HostView1D = Kokkos::View<T *, HostMemSpace>;

template <typename T> using MirrorView1D = typename View1D<T>::HostMirror;

// 2D Views
template <typename T> using View2D = Kokkos::View<T **, DefaultMemSpace>;

template <typename T> using HostView2D = Kokkos::View<T **, HostMemSpace>;

// Unmanaged views (for wrapping existing data)
template <typename T>
using UnmanagedView1D =
    Kokkos::View<T *, DefaultMemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

//=============================================================================
// Random number generator types
//=============================================================================

// Use Kokkos random number pool
using RNGPool = Kokkos::Random_XorShift64_Pool<DefaultExecSpace>;
using RNGState = typename RNGPool::generator_type;

//=============================================================================
// Atomic operations
//=============================================================================

template <typename T> KOKKOS_INLINE_FUNCTION void atomicAdd(T *ptr, T val) {
  Kokkos::atomic_add(ptr, val);
}

//=============================================================================
// Utility functions
//=============================================================================

KOKKOS_INLINE_FUNCTION
constexpr int mod(int x, int y) { return ((x % y) + y) % y; }

template <typename T> KOKKOS_INLINE_FUNCTION constexpr T pow2(T x) {
  return x * x;
}

template <typename T> KOKKOS_INLINE_FUNCTION T absVal(T x) {
  return (x >= 0) ? x : -x;
}

//=============================================================================
// Memory alignment for optimal performance
//=============================================================================

constexpr size_t KWQFT_ALIGNMENT = 64;

//=============================================================================
// Error handling
//=============================================================================

inline void checkError(const char *msg) {
  Kokkos::fence(); // Ensure all work is complete
                   // Additional error checking can be added here
}

//=============================================================================
// Print utilities
//=============================================================================

#ifdef KWQFT_VERBOSE
#define KWQFT_PRINT(...) printf(__VA_ARGS__)
#else
#define KWQFT_PRINT(...)                                                       \
  do {                                                                         \
  } while (0)
#endif

// Error message macro
#define KWQFT_ERROR(msg)                                                       \
  do {                                                                         \
    fprintf(stderr, "KWQFT Error: %s at %s:%d\n", msg, __FILE__, __LINE__);    \
    Kokkos::abort(msg);                                                        \
  } while (0)

// Warning message macro
#define KWQFT_WARNING(msg)                                                     \
  do {                                                                         \
    fprintf(stderr, "KWQFT Warning: %s\n", msg);                               \
  } while (0)

//=============================================================================
// Array type enumeration (for data layout)
//=============================================================================

enum class ArrayType {
  SOA,   // Structure of Arrays - full storage
  SOA12, // 12 parameter storage for SU(3)
  SOA8   // 8 parameter storage for SU(3)
};

//=============================================================================
// Read mode enumeration
//=============================================================================

enum class MemoryLocation {
  Device, // GPU or accelerator memory
  Host    // CPU memory
};

} // namespace kwqft

#endif // KWQFT_COMMON_HPP
