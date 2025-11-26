/**
 * @file gauge_array.hpp
 * @brief Gauge field array container for KWQFT
 * 
 * Provides a container for storing gauge field configurations
 * using Kokkos views for portable memory management
 */

#ifndef KWQFT_GAUGE_ARRAY_HPP
#define KWQFT_GAUGE_ARRAY_HPP

#include "kwqft_common.hpp"
#include "complex.hpp"
#include "matrixsun.hpp"
#include "constants.hpp"
#include "index.hpp"

namespace kwqft {

/**
 * @brief Gauge field array class
 * 
 * Stores gauge links as an array of complex numbers in SOA format
 * Each link is a NCOLORS x NCOLORS complex matrix
 * 
 * @tparam Real The underlying real type (float or double)
 */
template<typename Real>
class GaugeArray {
public:
    using ComplexT = Complex<Real>;
    using MatrixT = MatrixSun<Real, NCOLORS>;
    using ViewT = Kokkos::View<ComplexT*, DefaultMemSpace>;
    using host_ViewT = typename ViewT::HostMirror;

private:
    ViewT data_;            // Device data
    host_ViewT hostData_;  // Host mirror
    ArrayType arrayType_;   // Storage format
    MemoryLocation location_; // Where primary data resides
    bool evenOdd_;          // Even/odd ordering
    int size_;               // Number of links
    bool allocated_;         // Whether memory is allocated

public:
    // Default constructor
    GaugeArray() 
        : arrayType_(ArrayType::SOA), location_(MemoryLocation::Device),
          evenOdd_(false), size_(0), allocated_(false) {}

    // Constructor with parameters
    GaugeArray(ArrayType type, MemoryLocation loc, int size_in, bool even_odd = false)
        : arrayType_(type), location_(loc), evenOdd_(even_odd),
          size_(size_in), allocated_(false) {
        allocate(size_in);
    }

    // Destructor - Kokkos views handle cleanup automatically
    ~GaugeArray() = default;

    // Copy constructor (shallow copy of views)
    GaugeArray(const GaugeArray& other) = default;

    // Move constructor
    GaugeArray(GaugeArray&& other) = default;

    // Assignment operators
    GaugeArray& operator=(const GaugeArray& other) = default;
    GaugeArray& operator=(GaugeArray&& other) = default;

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
    int size() const { return size_; }

    KOKKOS_INLINE_FUNCTION
    bool allocated() const { return allocated_; }

    // Get raw data pointer (for kernels)
    KOKKOS_INLINE_FUNCTION
    ComplexT* data() { return data_.data(); }

    KOKKOS_INLINE_FUNCTION
    const ComplexT* data() const { return data_.data(); }

    // Get view
    ViewT& getView() { return data_; }
    const ViewT& getView() const { return data_; }

    // Get host view
    host_ViewT& getHostView() { return hostData_; }
    const host_ViewT& getHostView() const { return hostData_; }

    //=========================================================================
    // Memory management
    //=========================================================================

    /**
     * @brief Get number of complex elements per link
     */
    int getNumElems() const {
        switch (arrayType_) {
            case ArrayType::SOA:
                return NCOLORS * NCOLORS;
            case ArrayType::SOA12:
                return 6;
            case ArrayType::SOA8:
                return 4;
            default:
                return NCOLORS * NCOLORS;
        }
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
    float memoryMb() const {
        return bytes() / (1024.0f * 1024.0f);
    }

    /**
     * @brief Allocate memory
     */
    void allocate(int size_in) {
        if (allocated_) {
            KWQFT_WARNING("Array already allocated");
            return;
        }

        size_ = size_in;
        size_t total_elems = static_cast<size_t>(size_) * getNumElems();

        // Allocate device view
        data_ = ViewT("gauge_data", total_elems);
        
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
        if (!allocated_) return;
        Kokkos::deep_copy(data_, ComplexT::zero());
    }

    //=========================================================================
    // Data transfer
    //=========================================================================

    /**
     * @brief Copy data from device to host
     */
    void copyToHost() {
        Kokkos::deep_copy(hostData_, data_);
    }

    /**
     * @brief Copy data from host to device
     */
    void copyToDevice() {
        Kokkos::deep_copy(data_, hostData_);
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
        const ComplexT* ptr = data_.data();
        
        switch (arrayType_) {
            case ArrayType::SOA:
                for (int i = 0; i < NCOLORS; ++i) {
                    for (int j = 0; j < NCOLORS; ++j) {
                        m.e[i][j] = ptr[k + (j + i * NCOLORS) * size_];
                    }
                }
                break;
                
            case ArrayType::SOA12:
                // Store first two rows, reconstruct third
                for (int i = 0; i < NCOLORS - 1; ++i) {
                    for (int j = 0; j < NCOLORS; ++j) {
                        m.e[i][j] = ptr[k + (j + i * NCOLORS) * size_];
                    }
                }
                // Reconstruct third row for SU(3)
                if constexpr (NCOLORS == 3) {
                    m.e[2][0] = ~(m.e[0][1] * m.e[1][2] - m.e[0][2] * m.e[1][1]);
                    m.e[2][1] = ~(m.e[0][2] * m.e[1][0] - m.e[0][0] * m.e[1][2]);
                    m.e[2][2] = ~(m.e[0][0] * m.e[1][1] - m.e[0][1] * m.e[1][0]);
                }
                break;
                
            case ArrayType::SOA8:
                // 8-parameter reconstruction for SU(3)
                if constexpr (NCOLORS == 3) {
                    m.e[0][1] = ptr[k];
                    m.e[0][2] = ptr[k + size_];
                    m.e[1][0] = ptr[k + 2 * size_];
                    ComplexT theta = ptr[k + 3 * size_];
                    // Full reconstruction would go here
                    // This is simplified; full implementation needs reconstruct8p
                }
                break;
        }
        return m;
    }

    /**
     * @brief Set a matrix in the array at position k (device function)
     */
    KOKKOS_INLINE_FUNCTION
    void set(const MatrixT& A, int k) {
        ComplexT* ptr = data_.data();
        
        switch (arrayType_) {
            case ArrayType::SOA:
                for (int i = 0; i < NCOLORS; ++i) {
                    for (int j = 0; j < NCOLORS; ++j) {
                        ptr[k + (j + i * NCOLORS) * size_] = A.e[i][j];
                    }
                }
                break;
                
            case ArrayType::SOA12:
                for (int i = 0; i < NCOLORS - 1; ++i) {
                    for (int j = 0; j < NCOLORS; ++j) {
                        ptr[k + (j + i * NCOLORS) * size_] = A.e[i][j];
                    }
                }
                break;
                
            case ArrayType::SOA8:
                if constexpr (NCOLORS == 3) {
                    ptr[k] = A.e[0][1];
                    ComplexT theta;
                    theta.real() = A.e[0][0].phase();
                    theta.imag() = A.e[2][0].phase();
                    ptr[k + size_] = A.e[0][2];
                    ptr[k + 2 * size_] = A.e[1][0];
                    ptr[k + 3 * size_] = theta;
                }
                break;
        }
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
        const int num_elems = getNumElems();
        const ArrayType atype = arrayType_;

        Kokkos::parallel_for("GaugeArray::initCold",
            range_policy(0, size),
            KOKKOS_LAMBDA(const int k) {
                MatrixT I = MatrixT::identity();
                ComplexT* ptr = data_view.data();
                
                if (atype == ArrayType::SOA) {
                    for (int i = 0; i < NCOLORS; ++i) {
                        for (int j = 0; j < NCOLORS; ++j) {
                            ptr[k + (j + i * NCOLORS) * size] = I.e[i][j];
                        }
                    }
                } else if (atype == ArrayType::SOA12) {
                    for (int i = 0; i < NCOLORS - 1; ++i) {
                        for (int j = 0; j < NCOLORS; ++j) {
                            ptr[k + (j + i * NCOLORS) * size] = I.e[i][j];
                        }
                    }
                }
            }
        );
        Kokkos::fence();
    }

    /**
     * @brief Print array details
     */
    void details() const {
        const char* type_str = "SOA";
        if (arrayType_ == ArrayType::SOA12) type_str = "SOA12";
        else if (arrayType_ == ArrayType::SOA8) type_str = "SOA8";
        
        const char* loc_str = (location_ == MemoryLocation::Device) ? "Device" : "Host";
        const char* order_str = evenOdd_ ? "even/odd" : "normal";
        
        printf("GaugeArray: type=%s, location=%s, ordering=%s, size=%.2f MB\n",
               type_str, loc_str, order_str, memoryMb());
    }
};

// Type aliases
using gauges = GaugeArray<float>;
using gauged = GaugeArray<double>;

template<typename Real>
using gauge = GaugeArray<Real>;

} // namespace kwqft

#endif // KWQFT_GAUGE_ARRAY_HPP

