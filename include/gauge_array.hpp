/**
 * @file gauge_array.hpp
 * @brief Gauge field array container for KWQFT
 *
 * Provides a container for storing gauge field configurations
 * using Kokkos views for portable memory management
 */

#ifndef KWQFT_GAUGE_ARRAY_HPP
#define KWQFT_GAUGE_ARRAY_HPP

#include "complex.hpp"
#include "constants.hpp"
#include "gauge_halo.hpp"
#include "gauge_load_save.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"

#include <memory>

namespace kwqft {

/**
 * @brief Gauge field array class
 *
 * Stores gauge links as an array of complex numbers in SOA format
 * Each link is a NCOLORS x NCOLORS complex matrix
 *
 * @tparam Real The underlying real type (float or double)
 */
template <typename Real> class GaugeArray {
public:
  using ComplexT = Complex<Real>;
  using MatrixT = MatrixSun<Real, NCOLORS>;
  using ViewT = Kokkos::View<ComplexT *, DefaultMemSpace>;
  using HostViewT = typename ViewT::host_mirror_type;

private:
  ViewT m_data;              // Device data
  HostViewT host_data;     // Host mirror
  ArrayType array_type;     // Storage format
  MemoryLocation m_location; // Where primary data resides
  bool even_odd;            // Even/odd ordering (required; must be true)
  int64_t m_size;            // Number of links (int64_t for large lattices)
  bool m_allocated;          // Whether memory is allocated
  /// MPI ghost buffers shared by every kernel operating on this field
  /// (created lazily; null when not domain-decomposed).
  std::shared_ptr<GaugeHaloBuffers<Real>> m_halo;

public:
  // Default constructor
  GaugeArray()
      : array_type(ArrayType::SOA), m_location(MemoryLocation::Device),
        even_odd(false), m_size(0), m_allocated(false) {}

  // Constructor with parameters
  GaugeArray(ArrayType type, MemoryLocation loc, int64_t size_in,
             bool even_odd = true)
      : array_type(type), m_location(loc), even_odd(even_odd), m_size(size_in),
        m_allocated(false) {
    allocate(size_in);
  }

  // Destructor - Kokkos views handle cleanup automatically
  ~GaugeArray() = default;

  // Copy constructor (shallow copy of views)
  GaugeArray(const GaugeArray &other) = default;

  // Move constructor
  GaugeArray(GaugeArray &&other) = default;

  // Assignment operators
  GaugeArray &operator=(const GaugeArray &other) = default;
  GaugeArray &operator=(GaugeArray &&other) = default;

  //=========================================================================
  // Accessors
  //=========================================================================

  KOKKOS_INLINE_FUNCTION
  ArrayType type() const { return array_type; }

  KOKKOS_INLINE_FUNCTION
  MemoryLocation location() const { return m_location; }

  KOKKOS_INLINE_FUNCTION
  bool evenOdd() const { return even_odd; }

  KOKKOS_INLINE_FUNCTION
  int64_t size() const { return m_size; }

  KOKKOS_INLINE_FUNCTION
  bool allocated() const { return m_allocated; }

  // Get raw data pointer (for kernels)
  KOKKOS_INLINE_FUNCTION
  ComplexT *data() { return m_data.data(); }

  KOKKOS_INLINE_FUNCTION
  const ComplexT *data() const { return m_data.data(); }

  // Get view
  ViewT &getView() { return m_data; }
  const ViewT &getView() const { return m_data; }

  // Get host view
  HostViewT &getHostView() { return host_data; }
  const HostViewT &getHostView() const { return host_data; }

  //=========================================================================
  // MPI halo (shared across kernels)
  //=========================================================================

  /**
   * @brief Shared halo buffers for \p params, or nullptr when not decomposed.
   *
   * All update/measurement kernels use this single instance so that ghost
   * data sent by one kernel is reused by the next.
   */
  GaugeHaloBuffers<Real> *halo(const LatticeParams &params) {
    if (!params.mpi || params.nproc <= 1) {
      return nullptr;
    }
    if (!m_halo) {
      m_halo = std::make_shared<GaugeHaloBuffers<Real>>(params);
    }
    return m_halo.get();
  }

  /// Call after any whole-field write that bypasses the halo bookkeeping.
  void invalidateHalo() {
    if (m_halo) {
      m_halo->invalidate();
    }
  }

  //=========================================================================
  // Memory management
  //=========================================================================

  /**
   * @brief Get number of complex elements per link
   */
  int getNumElems() const { return gaugeComplexElems(array_type); }

  /**
   * @brief Get total memory size in bytes
   */
  size_t bytes() const {
    return static_cast<size_t>(m_size) * getNumElems() * sizeof(ComplexT);
  }

  /**
   * @brief Get memory size in MB
   */
  float memoryMb() const { return bytes() / (1024.0f * 1024.0f); }

  /**
   * @brief Allocate memory
   */
  void allocate(int64_t size_in) {
    if (m_allocated) {
      KWQFT_WARNING("Array already allocated");
      return;
    }
    if (!even_odd) {
      KWQFT_ERROR("GaugeArray requires even/odd (checkerboard) ordering");
    }

    m_size = size_in;
    size_t total_elems = static_cast<size_t>(m_size) * getNumElems();

    // Allocate device view
    m_data = ViewT("gauge_data", total_elems);

    // Create host mirror
    host_data = Kokkos::create_mirror_view(m_data);

    m_allocated = true;
  }

  /**
   * @brief Release memory (Kokkos handles this automatically)
   */
  void release() {
    if (m_allocated) {
      m_data = ViewT();
      host_data = HostViewT();
      m_halo.reset();
      m_size = 0;
      m_allocated = false;
    }
  }

  /**
   * @brief Zero out the data
   */
  void clean() {
    if (!m_allocated)
      return;
    Kokkos::deep_copy(m_data, ComplexT::zero());
    invalidateHalo();
  }

  //=========================================================================
  // Data transfer
  //=========================================================================

  /**
   * @brief Copy data from device to host
   */
  void copyToHost() { Kokkos::deep_copy(host_data, m_data); }

  /**
   * @brief Copy data from host to device
   */
  void copyToDevice() {
    Kokkos::deep_copy(m_data, host_data);
    invalidateHalo();
  }

  //=========================================================================
  // Matrix access functions (for kernels)
  //=========================================================================

  /**
   * @brief Get a matrix from the array at position k (device function)
   */
  KOKKOS_INLINE_FUNCTION
  MatrixT get(int k) const {
    MatrixT m;
    loadGaugeMatrix(m_data.data(), static_cast<int64_t>(k), m_size, array_type,
                    m);
    return m;
  }

  /**
   * @brief Set a matrix in the array at position k (device function)
   */
  KOKKOS_INLINE_FUNCTION
  void set(const MatrixT &a, int k) {
    storeGaugeMatrix(m_data.data(), static_cast<int64_t>(k), m_size, array_type,
                     a);
  }

  //=========================================================================
  // Initialization
  //=========================================================================

  /**
   * @brief Initialize with cold start (identity matrices)
   */
  void initCold() {
    const int size = m_size;
    auto data_view = m_data;
    const ArrayType atype = array_type;

    Kokkos::parallel_for(
        "GaugeArray::initCold", RangePolicy(0, size),
        KOKKOS_LAMBDA(const int k) {
          MatrixT I = MatrixT::identity();
          storeGaugeMatrix(data_view.data(), static_cast<int64_t>(k),
                           static_cast<int64_t>(size), atype, I);
        });
    Kokkos::fence();
    invalidateHalo();
  }

  /**
   * @brief Print array details
   */
  void details() const {
    const char *type_str = "SOA";
    if (array_type == ArrayType::SOA12)
      type_str = "SOA12";
    else if (array_type == ArrayType::SOA8)
      type_str = "SOA8";

    const char *loc_str =
        (m_location == MemoryLocation::Device) ? "Device" : "Host";
    const char *order_str = even_odd ? "even/odd" : "normal";

    printf("GaugeArray: type=%s, location=%s, ordering=%s, size=%.2f MB\n",
           type_str, loc_str, order_str, memoryMb());
  }
};

// Type aliases
using Gauges = GaugeArray<float>;
using Gauged = GaugeArray<double>;

template <typename Real> using Gauge = GaugeArray<Real>;

} // namespace kwqft

#endif // KWQFT_GAUGE_ARRAY_HPP
