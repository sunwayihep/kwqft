/**
 * @file index.hpp
 * @brief Lattice indexing functions for KWQFT
 *
 * Provides functions for converting between different lattice index schemes
 * including normal ordering and even/odd (checkerboard) ordering
 */

#ifndef KWQFT_INDEX_HPP
#define KWQFT_INDEX_HPP

#include "constants.hpp"
#include "kwqft_common.hpp"

namespace kwqft {

//=============================================================================
// Normal ordering index functions
//=============================================================================

/**
 * @brief Convert 1D index to N-dimensional coordinates
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION void indexNdNm(int64_t id, int x[ND],
                                      const LatticeParams &p) {
  int64_t temp = id;
  for (int i = 0; i < ND; ++i) {
    x[i] = static_cast<int>(temp % p.grid[i]);
    temp /= p.grid[i];
  }
}

/**
 * @brief Convert 1D index to N-dimensional coordinates with custom grid
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION void indexNdNm(int64_t id, int x[ND], const int X[ND]) {
  int64_t temp = id;
  for (int i = 0; i < ND; ++i) {
    x[i] = static_cast<int>(temp % X[i]);
    temp /= X[i];
  }
}

/**
 * @brief Convert N-dimensional coordinates to 1D index
 * Returns int64_t to support large lattices
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNm(const int x[ND],
                                         const LatticeParams &p) {
  int64_t index = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND; ++i) {
    index += x[i] * factor;
    factor *= p.grid[i];
  }
  return index;
}

/**
 * @brief Convert N-dimensional coordinates to 1D index with custom grid
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNm(const int x[ND], const int X[ND]) {
  int64_t index = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND; ++i) {
    index += x[i] * factor;
    factor *= X[i];
  }
  return index;
}

/**
 * @brief Get neighbor index in normal ordering with periodic boundary
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNeigNm(int64_t id, int mu, int lmu,
                                             const LatticeParams &p) {
  int x[ND];
  indexNdNm<ND>(id, x, p);
  x[mu] = (x[mu] + lmu + p.grid[mu]) % p.grid[mu];
  return indexNdNm<ND>(x, p);
}

/**
 * @brief Get neighbor index with two direction shifts
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNeigNm(int64_t id, int mu, int lmu,
                                             int nu, int lnu,
                                             const LatticeParams &p) {
  int x[ND];
  indexNdNm<ND>(id, x, p);
  x[mu] = (x[mu] + lmu + p.grid[mu]) % p.grid[mu];
  x[nu] = (x[nu] + lnu + p.grid[nu]) % p.grid[nu];
  return indexNdNm<ND>(x, p);
}

//=============================================================================
// Even/Odd (Checkerboard) ordering index functions
//=============================================================================

/**
 * @brief Convert even/odd index to N-dimensional coordinates
 * @param id Even/odd index (0 to volume/2 - 1)
 * @param x Output coordinates
 * @param oddbit 0 for even sites, 1 for odd sites
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION void indexNdEo(int x[ND], int64_t id, int oddbit,
                                      const LatticeParams &p) {
  int64_t factor = id / (p.grid[0] / 2);
  for (int i = 1; i < ND; ++i) {
    int64_t factor1 = factor / p.grid[i];
    x[i] = static_cast<int>(factor - factor1 * p.grid[i]);
    factor = factor1;
  }
  int sum = 0;
  for (int i = 1; i < ND; ++i) {
    sum += x[i];
  }
  int xodd = (sum + oddbit) & 1;
  x[0] = static_cast<int>((id * 2 + xodd) - id / (p.grid[0] / 2) * p.grid[0]);
}

/**
 * @brief Convert even/odd index to N-dimensional coordinates with custom grid
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION void indexNdEo(int x[ND], int64_t id, int oddbit,
                                      const int X[ND]) {
  int64_t factor = id / (X[0] / 2);
  for (int i = 1; i < ND; ++i) {
    int64_t factor1 = factor / X[i];
    x[i] = static_cast<int>(factor - factor1 * X[i]);
    factor = factor1;
  }
  int sum = 0;
  for (int i = 1; i < ND; ++i) {
    sum += x[i];
  }
  int xodd = (sum + oddbit) & 1;
  x[0] = static_cast<int>((id * 2 + xodd) - id / (X[0] / 2) * X[0]);
}

/**
 * @brief Get neighbor index in even/odd ordering
 * Returns the index in the half-volume array with parity offset
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNeigEo(int64_t id, int oddbit, int mu,
                                             int lmu, const LatticeParams &p) {
  int x[ND];
  indexNdEo<ND>(x, id, oddbit, p);

  // Move to neighbor (with periodic boundary)
  x[mu] = (x[mu] + lmu + p.grid[mu]) % p.grid[mu];

  // Convert coordinates to normal linear index, then to EO index
  int64_t pos = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND; ++i) {
    pos += x[i] * factor;
    factor *= p.grid[i];
  }
  pos /= 2; // Convert to half-volume index

  // Determine parity of neighbor and add offset for odd sites
  int sum_x = 0;
  for (int i = 0; i < ND; ++i) {
    sum_x += x[i];
  }
  int oddbit1 = sum_x & 1;
  pos += oddbit1 * p.half_volume;

  return pos;
}

/**
 * @brief Get neighbor index with two direction shifts in even/odd ordering
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNeigEo(int64_t id, int oddbit, int mu,
                                             int lmu, int nu, int lnu,
                                             const LatticeParams &p) {
  int x[ND];
  indexNdEo<ND>(x, id, oddbit, p);

  // Move to neighbor in both directions (with periodic boundary)
  x[mu] = (x[mu] + lmu + p.grid[mu]) % p.grid[mu];
  x[nu] = (x[nu] + lnu + p.grid[nu]) % p.grid[nu];

  // Convert coordinates to normal linear index, then to EO index
  int64_t pos = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND; ++i) {
    pos += x[i] * factor;
    factor *= p.grid[i];
  }
  pos /= 2; // Convert to half-volume index

  // Determine parity of neighbor and add offset for odd sites
  int sum_x = 0;
  for (int i = 0; i < ND; ++i) {
    sum_x += x[i];
  }
  int oddbit1 = sum_x & 1;
  pos += oddbit1 * p.half_volume;

  return pos;
}

/**
 * @brief Get neighbor index using coordinate array
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNeigEo(const int y[ND], int mu, int lmu,
                                             const LatticeParams &p) {
  int x[ND];
  for (int i = 0; i < ND; ++i) {
    x[i] = y[i];
  }
  x[mu] = (x[mu] + lmu + p.grid[mu]) % p.grid[mu];

  // Convert coordinates to normal linear index, then to EO index
  int64_t pos = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND; ++i) {
    pos += x[i] * factor;
    factor *= p.grid[i];
  }
  pos /= 2;

  // Determine parity and add offset for odd sites
  int sum_x = 0;
  for (int i = 0; i < ND; ++i) {
    sum_x += x[i];
  }
  int oddbit1 = sum_x & 1;
  pos += oddbit1 * p.half_volume;

  return pos;
}

/**
 * @brief Get neighbor +1 in direction mu (optimized)
 *
 * The neighbor of an even site in +mu direction is always odd (and vice versa)
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNeigEoPlusOne(int64_t id, int oddbit,
                                                    int mu,
                                                    const LatticeParams &p) {
  int x[ND];
  indexNdEo<ND>(x, id, oddbit, p);

  x[mu] = (x[mu] + 1) % p.grid[mu];

  // Convert to normal index and divide by 2
  int64_t pos = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND; ++i) {
    pos += x[i] * factor;
    factor *= p.grid[i];
  }
  pos /= 2;

  // Neighbor has opposite parity
  pos += (1 - oddbit) * p.half_volume;

  return pos;
}

/**
 * @brief Get neighbor -1 in direction mu (optimized)
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdNeigEoMinusOne(int64_t id, int oddbit,
                                                     int mu,
                                                     const LatticeParams &p) {
  int x[ND];
  indexNdEo<ND>(x, id, oddbit, p);

  x[mu] = (x[mu] - 1 + p.grid[mu]) % p.grid[mu];

  // Convert to normal index and divide by 2
  int64_t pos = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND; ++i) {
    pos += x[i] * factor;
    factor *= p.grid[i];
  }
  pos /= 2;

  // Neighbor has opposite parity
  pos += (1 - oddbit) * p.half_volume;

  return pos;
}

//=============================================================================
// Spatial index functions (NDIMS - 1 dimensions)
//=============================================================================

/**
 * @brief Convert 1D spatial index to (NDIMS-1)-dimensional coordinates
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION void indexNdsNm(int64_t id, int x[ND - 1],
                                       const LatticeParams &p) {
  int64_t temp = id;
  for (int i = 0; i < ND - 1; ++i) {
    x[i] = static_cast<int>(temp % p.grid[i]);
    temp /= p.grid[i];
  }
}

/**
 * @brief Convert (NDIMS-1)-dimensional coordinates to 1D spatial index
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdsNm(const int x[ND - 1],
                                          const LatticeParams &p) {
  int64_t index = 0;
  int64_t factor = 1;
  for (int i = 0; i < ND - 1; ++i) {
    index += x[i] * factor;
    factor *= p.grid[i];
  }
  return index;
}

/**
 * @brief Get spatial neighbor index
 */
template <int ND = NDIMS>
KOKKOS_INLINE_FUNCTION int64_t indexNdsNeigNm(int64_t id, int mu, int r,
                                              const LatticeParams &p) {
  int x[ND - 1];
  indexNdsNm<ND>(id, x, p);
  x[mu] = (x[mu] + r + p.grid[mu]) % p.grid[mu];
  return indexNdsNm<ND>(x, p);
}

} // namespace kwqft

#endif // KWQFT_INDEX_HPP
