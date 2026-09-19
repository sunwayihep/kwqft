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
#include "gauge_load_save.hpp"
#include "index.hpp"
#include "kwqft_common.hpp"
#include "matrixsun.hpp"

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
  using host_ViewT = typename ViewT::host_mirror_type;

private:
  ViewT data_;              // Device data
  host_ViewT hostData_;     // Host mirror
  ArrayType arrayType_;     // Storage format
  MemoryLocation location_; // Where primary data resides
  bool evenOdd_;            // Even/odd ordering (required; must be true)
  int64_t size_;            // Number of links (int64_t for large lattices)
  bool allocated_;          // Whether memory is allocated

public:
  // Default constructor
  GaugeArray()
      : arrayType_(ArrayType::SOA), location_(MemoryLocation::Device),
        evenOdd_(false), size_(0), allocated_(false) {}

  // Constructor with parameters
  GaugeArray(ArrayType type, MemoryLocation loc, int64_t sizeIn,
             bool evenOdd = true)
      : arrayType_(type), location_(loc), evenOdd_(evenOdd), size_(sizeIn),
        allocated_(false) {
    allocate(sizeIn);
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
  ArrayType type() const { return arrayType_; }

  KOKKOS_INLINE_FUNCTION
  MemoryLocation location() const { return location_; }

  KOKKOS_INLINE_FUNCTION
  bool even_odd() const { return evenOdd_; }

  KOKKOS_INLINE_FUNCTION
  int64_t size() const { return size_; }

  KOKKOS_INLINE_FUNCTION
  bool allocated() const { return allocated_; }

  // Get raw data pointer (for kernels)
  KOKKOS_INLINE_FUNCTION
  ComplexT *data() { return data_.data(); }

  KOKKOS_INLINE_FUNCTION
  const ComplexT *data() const { return data_.data(); }

  // Get view
  ViewT &getView() { return data_; }
  const ViewT &getView() const { return data_; }

  // Get host view
  host_ViewT &getHostView() { return hostData_; }
  const host_ViewT &getHostView() const { return hostData_; }

  //=========================================================================
  // Memory management
  //=========================================================================

  /**
   * @brief Get number of complex elements per link
   */
  int getNumElems() const {
    return gauge_complex_elems(arrayType_);
  }

  /**
   * @brief Get total memory size in bytes
   */
  size_t bytes() const {
    return static_cast<size_t>(size_) * getNumElems() * sizeof(ComplexT);
  }

  /**
   * @brief Get memory size in MB
   */
  float memoryMb() const { return bytes() / (1024.0f * 1024.0f); }

  /**
   * @brief Allocate memory
   */
  void allocate(int64_t sizeIn) {
    if (allocated_) {
      KWQFT_WARNING("Array already allocated");
      return;
    }
    if (!evenOdd_) {
      KWQFT_ERROR("GaugeArray requires even/odd (checkerboard) ordering");
    }

    size_ = sizeIn;
    size_t totalElems = static_cast<size_t>(size_) * getNumElems();

    // Allocate device view
    data_ = ViewT("gauge_data", totalElems);

    // Create host mirror
    hostData_ = Kokkos::create_mirror_view(data_);

    allocated_ = true;
  }

  /**
   * @brief Release memory (Kokkos handles this automatically)
   */
  void release() {
    if (allocated_) {
      data_ = ViewT();
      hostData_ = host_ViewT();
      size_ = 0;
      allocated_ = false;
    }
  }

  /**
   * @brief Zero out the data
   */
  void clean() {
    if (!allocated_)
      return;
    Kokkos::deep_copy(data_, ComplexT::zero());
  }

  //=========================================================================
  // Data transfer
  //=========================================================================

  /**
   * @brief Copy data from device to host
   */
  void copyToHost() { Kokkos::deep_copy(hostData_, data_); }

  /**
   * @brief Copy data from host to device
   */
  void copyToDevice() { Kokkos::deep_copy(data_, hostData_); }

  //=========================================================================
  // Matrix access functions (for kernels)
  //=========================================================================

  /**
   * @brief Get a matrix from the array at position k (device function)
   */
  KOKKOS_INLINE_FUNCTION
  MatrixT get(int k) const {
    MatrixT m;
    loadGaugeMatrix(data_.data(), static_cast<int64_t>(k), size_, arrayType_,
                    m);
    return m;
  }

  /**
   * @brief Set a matrix in the array at position k (device function)
   */
  KOKKOS_INLINE_FUNCTION
  void set(const MatrixT &A, int k) {
    storeGaugeMatrix(data_.data(), static_cast<int64_t>(k), size_, arrayType_,
                     A);
  }

  //=========================================================================
  // Initialization
  //=========================================================================

  /**
   * @brief Initialize with cold start (identity matrices)
   */
  void initCold() {
    const int size = size_;
    auto data_view = data_;
    const ArrayType atype = arrayType_;

    Kokkos::parallel_for(
        "GaugeArray::initCold", range_policy(0, size),
        KOKKOS_LAMBDA(const int k) {
          MatrixT I = MatrixT::identity();
          storeGaugeMatrix(data_view.data(), static_cast<int64_t>(k),
                           static_cast<int64_t>(size), atype, I);
        });
    Kokkos::fence();
  }

  /**
   * @brief Print array details
   */
  void details() const {
    const char *type_str = "SOA";
    if (arrayType_ == ArrayType::SOA12)
      type_str = "SOA12";
    else if (arrayType_ == ArrayType::SOA8)
      type_str = "SOA8";

    const char *loc_str =
        (location_ == MemoryLocation::Device) ? "Device" : "Host";
    const char *order_str = evenOdd_ ? "even/odd" : "normal";

    printf("GaugeArray: type=%s, location=%s, ordering=%s, size=%.2f MB\n",
           type_str, loc_str, order_str, memoryMb());
  }
};

// Type aliases
using gauges = GaugeArray<float>;
using gauged = GaugeArray<double>;

template <typename Real> using gauge = GaugeArray<Real>;

} // namespace kwqft

#endif // KWQFT_GAUGE_ARRAY_HPP
