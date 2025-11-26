/**
 * @file io_gauge.cpp
 * @brief I/O functions for gauge field configurations
 */

#include "gauge_array.hpp"
#include <fstream>
#include <string>

namespace kwqft {

/**
 * @brief Save gauge configuration to binary file
 */
template<typename Real>
void save_gauge_binary(const GaugeArray<Real>& gauge, const std::string& filename) {
    // Create host mirror
    auto host_view = Kokkos::create_mirror_view(gauge.getView());
    Kokkos::deep_copy(host_view, gauge.getView());
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        KWQFT_ERROR("Cannot open file for writing");
        return;
    }
    
    // Write header
    int ncolors = NCOLORS;
    int ndims = NDIMS;
    int size = gauge.size();
    int num_elems = gauge.getNumElems();
    
    file.write(reinterpret_cast<char*>(&ncolors), sizeof(int));
    file.write(reinterpret_cast<char*>(&ndims), sizeof(int));
    file.write(reinterpret_cast<char*>(&size), sizeof(int));
    file.write(reinterpret_cast<char*>(&num_elems), sizeof(int));
    
    // Write data
    size_t total_size = static_cast<size_t>(size) * num_elems * sizeof(Complex<Real>);
    file.write(reinterpret_cast<const char*>(host_view.data()), total_size);
    
    file.close();
    printf("Saved gauge configuration to %s (%.2f MB)\n", 
           filename.c_str(), total_size / (1024.0 * 1024.0));
}

/**
 * @brief Load gauge configuration from binary file
 */
template<typename Real>
void load_gauge_binary(GaugeArray<Real>& gauge, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        KWQFT_ERROR("Cannot open file for reading");
        return;
    }
    
    // Read header
    int ncolors, ndims, size, num_elems;
    file.read(reinterpret_cast<char*>(&ncolors), sizeof(int));
    file.read(reinterpret_cast<char*>(&ndims), sizeof(int));
    file.read(reinterpret_cast<char*>(&size), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_elems), sizeof(int));
    
    // Verify compatibility
    if (ncolors != NCOLORS || ndims != NDIMS) {
        KWQFT_ERROR("Gauge file has incompatible NCOLORS or NDIMS");
        return;
    }
    
    if (size != gauge.size()) {
        KWQFT_ERROR("Gauge file has different size");
        return;
    }
    
    // Read data to host
    auto host_view = Kokkos::create_mirror_view(gauge.getView());
    size_t total_size = static_cast<size_t>(size) * num_elems * sizeof(Complex<Real>);
    file.read(reinterpret_cast<char*>(host_view.data()), total_size);
    
    // Copy to device
    Kokkos::deep_copy(gauge.getView(), host_view);
    
    file.close();
    printf("Loaded gauge configuration from %s\n", filename.c_str());
}

// Explicit instantiations
template void save_gauge_binary<float>(const GaugeArray<float>&, const std::string&);
template void save_gauge_binary<double>(const GaugeArray<double>&, const std::string&);
template void load_gauge_binary<float>(GaugeArray<float>&, const std::string&);
template void load_gauge_binary<double>(GaugeArray<double>&, const std::string&);

} // namespace kwqft

