/**
 * @file measurements.hpp
 * @brief Measurement observables for KWQFT
 * 
 * Implements plaquette and Polyakov loop measurements
 * using Kokkos parallel reductions
 */

#ifndef KWQFT_MEASUREMENTS_HPP
#define KWQFT_MEASUREMENTS_HPP

#include "kwqft_common.hpp"
#include "complex.hpp"
#include "matrixsun.hpp"
#include "gauge_array.hpp"
#include "constants.hpp"
#include "index.hpp"

namespace kwqft {

//=============================================================================
// Plaquette measurement
//=============================================================================

/**
 * @brief Calculate plaquette expectation value
 * 
 * Computes the average plaquette = (1/Nc) * Re Tr(U_plaq)
 * where U_plaq = U_mu(x) * U_nu(x+mu) * U_mu^\dagger(x+nu) * U_nu^\dagger(x)
 * 
 * Uses even-odd (checkerboard) storage format
 */
template<typename Real>
class Plaquette {
public:
    using GaugeT = GaugeArray<Real>;
    using MatrixT = MatrixSun<Real, NCOLORS>;
    using ComplexT = Complex<Real>;

private:
    GaugeT& gauge_;
    LatticeParams params_;
    Real plaqValue_;
    Real spatialValue_;
    Real temporalValue_;
    double time_;

    /**
     * @brief Helper function to load a gauge link from even-odd storage
     */
    KOKKOS_INLINE_FUNCTION
    static void loadLink(MatrixT& U, const ComplexT* gauge_ptr, 
                          int idx, int mu, int volume, int size) {
        int muvolume = mu * volume;
        for (int i = 0; i < NCOLORS; ++i) {
            for (int j = 0; j < NCOLORS; ++j) {
                U.e[i][j] = gauge_ptr[idx + muvolume + (j + i * NCOLORS) * size];
            }
        }
    }

    /**
     * @brief Get neighbor index in even-odd ordering (simplified for single GPU)
     * 
     */
    KOKKOS_INLINE_FUNCTION
    static int getNeighborEo(int id, int oddbit, int mu, int lmu, 
                               const LatticeParams& p) {
        // Convert even-odd index to coordinates using indexNdEo logic
        int x[NDIMS];
        int factor = id / (p.grid[0] / 2);
        for (int i = 1; i < NDIMS; ++i) {
            int factor1 = factor / p.grid[i];
            x[i] = factor - factor1 * p.grid[i];
            factor = factor1;
        }
        int sum = 0;
        for (int i = 1; i < NDIMS; ++i) {
            sum += x[i];
        }
        int xodd = (sum + oddbit) & 1;
        x[0] = (id * 2 + xodd) - id / (p.grid[0] / 2) * p.grid[0];
        
        // Move to neighbor
        x[mu] = (x[mu] + lmu + p.grid[mu]) % p.grid[mu];
        
        // Convert coordinates to normal linear index, then to EO index
        // pos = indexNdNm(x) / 2
        int pos = 0;
        factor = 1;
        for (int i = 0; i < NDIMS; ++i) {
            pos += x[i] * factor;
            factor *= p.grid[i];
        }
        pos /= 2;  // Convert to half-volume index
        
        // Determine parity of neighbor and add offset for odd sites
        int sum_x = 0;
        for (int i = 0; i < NDIMS; ++i) {
            sum_x += x[i];
        }
        int oddbit1 = sum_x & 1;
        pos += oddbit1 * p.half_volume;
        
        return pos;
    }

public:
    Plaquette(GaugeT& gauge, const LatticeParams& params)
        : gauge_(gauge), params_(params), 
          plaqValue_(0), spatialValue_(0), temporalValue_(0), time_(0) {}

    /**
     * @brief Compute the plaquette using even-odd storage format
     */
    void run() {
        Kokkos::Timer timer;
        
        auto gauge_view = gauge_.getView();
        auto params = params_;
        int size = gauge_.size();
        int volume = params.volume;
        int half_vol = params.half_volume;
        
        Real plaq_sum = 0;
        Real spatial_sum = 0;
        Real temporal_sum = 0;
        
        // Parallel reduction over all sites (even + odd)
        Kokkos::parallel_reduce("Plaquette",
            range_policy(0, volume),
            KOKKOS_LAMBDA(const int idd, Real& lsum, Real& ssum, Real& tsum) {
                ComplexT* gauge_ptr = gauge_view.data();
                
                // Determine parity and half-volume index
                int oddbit = 0;
                int id = idd;
                if (idd >= half_vol) {
                    oddbit = 1;
                    id = idd - half_vol;
                }
                
                // Current site index in even-odd storage
                int idxoddbit = id + oddbit * half_vol;
                
                // Calculate plaquettes for all mu < nu directions
                MatrixT link1, link;
                
                for (int mu = 0; mu < NDIMS; ++mu) {
                    // Load U_mu(x)
                    loadLink(link1, gauge_ptr, idxoddbit, mu, volume, size);
                    
                    // Get neighbor index x + mu
                    int newidmu1 = getNeighborEo(id, oddbit, mu, 1, params);
                    
                    for (int nu = mu + 1; nu < NDIMS; ++nu) {
                        // Load U_nu(x+mu)
                        MatrixT U_nu_xmu;
                        loadLink(U_nu_xmu, gauge_ptr, newidmu1, nu, volume, size);
                        
                        // Load U_mu(x+nu) - dagger
                        int newidnu1 = getNeighborEo(id, oddbit, nu, 1, params);
                        MatrixT U_mu_xnu;
                        loadLink(U_mu_xnu, gauge_ptr, newidnu1, mu, volume, size);
                        
                        // Load U_nu(x) - dagger
                        MatrixT U_nu;
                        loadLink(U_nu, gauge_ptr, idxoddbit, nu, volume, size);
                        
                        // Compute U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
                        link = U_nu_xmu * U_mu_xnu.dagger() * U_nu.dagger();
                        Real tr = (link1 * link).realtrace();
                        
                        // Separate spatial and temporal plaquettes
                        if (nu == NDIMS - 1) {
                            tsum += tr;  // Temporal plaquette (mu,T) or (T,nu)
                        } else {
                            ssum += tr;  // Spatial plaquette
                        }
                    }
                }
            },
            plaq_sum, spatial_sum, temporal_sum
        );
        
        // Normalize: divide by Nc and number of plaquettes
        // Average over different spatial and time directions
        spatialValue_ = spatial_sum / (Real(NCOLORS) * volume * TOTAL_NUM_SPLAQS);
        temporalValue_ = temporal_sum / (Real(NCOLORS) * volume * TOTAL_NUM_TPLAQS);
        plaqValue_ = (spatialValue_ + temporalValue_) / Real(2);
        
        time_ = timer.seconds();
    }

    Real value() const { return plaqValue_; }
    Real spatial() const { return spatialValue_; }
    Real temporal() const { return temporalValue_; }
    double time() const { return time_; }

    /**
     * @brief Calculate number of floating point operations
     * 
     * The factor 120 includes all plaquette directions per site
     */
    long long flop() const {
        return static_cast<long long>(NCOLORS) * NCOLORS * NCOLORS * 120LL 
               * params_.volume;
    }

    /**
     * @brief Calculate bytes read
     * 
     */
    long long bytes() const {
        int nuparams_ = NCOLORS * NCOLORS * 2;  // SOA format (real + imag)
        return (22LL * nuparams_ + 4LL) * params_.volume * sizeof(Real);
    }

    double flops() const {
        return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
    }

    double bandwidth() const {
        return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30)) : 0.0;
    }

    void printValue() const {
        printf("Plaquette: %.12f (spatial: %.12f, temporal: %.12f)\n",
               static_cast<double>(plaqValue_),
               static_cast<double>(spatialValue_),
               static_cast<double>(temporalValue_));
    }

    void stat() const {
        printf("Plaquette:  %.4f s\t%.2f GB/s\t%.2f GFlops\n", 
               time_, bandwidth(), flops());
    }
};

//=============================================================================
// Polyakov loop measurement
//=============================================================================

/**
 * @brief Calculate Polyakov loop
 * 
 * The Polyakov loop is the trace of the product of temporal links
 * at a fixed spatial position
 */
template<typename Real>
class PolyakovLoop {
public:
    using GaugeT = GaugeArray<Real>;
    using MatrixT = MatrixSun<Real, NCOLORS>;
    using ComplexT = Complex<Real>;

private:
    GaugeT& gauge_;
    LatticeParams params_;
    ComplexT polyValue_;
    double time_;

public:
    PolyakovLoop(GaugeT& gauge, const LatticeParams& params)
        : gauge_(gauge), params_(params), polyValue_(0, 0), time_(0) {}

    /**
     * @brief Compute the Polyakov loop
     */
    void run() {
        Kokkos::Timer timer;
        
        auto gauge_view = gauge_.getView();
        auto params = params_;
        int size = gauge_.size();
        
        // Calculate spatial volume
        int spatial_volume = 1;
        for (int i = 0; i < NDIMS - 1; ++i) {
            spatial_volume *= params.grid[i];
        }
        
        int Nt = params.grid[NDIMS - 1];
        int volume = params.volume;
        int t_dir = NDIMS - 1;
        int t_volume = t_dir * volume;
        
        Real poly_re = 0;
        Real poly_im = 0;
        
        // Parallel reduction over spatial sites
        Kokkos::parallel_reduce("PolyakovLoop",
            range_policy(0, spatial_volume),
            KOKKOS_LAMBDA(const int spatial_idx, Real& re_sum, Real& im_sum) {
                ComplexT* gauge_ptr = gauge_view.data();
                
                // Convert spatial index to coordinates
                int x[NDIMS];
                int temp = spatial_idx;
                for (int i = 0; i < NDIMS - 1; ++i) {
                    x[i] = temp % params.grid[i];
                    temp /= params.grid[i];
                }
                x[NDIMS - 1] = 0;  // Start at t=0
                
                // Product of temporal links
                MatrixT poly = MatrixT::identity();
                
                for (int t = 0; t < Nt; ++t) {
                    x[NDIMS - 1] = t;
                    int idx = indexNdNm<NDIMS>(x, params);
                    
                    // Load temporal link
                    MatrixT U_t;
                    for (int i = 0; i < NCOLORS; ++i) {
                        for (int j = 0; j < NCOLORS; ++j) {
                            U_t.e[i][j] = gauge_ptr[idx + t_volume + 
                                         (j + i * NCOLORS) * size];
                        }
                    }
                    
                    poly = poly * U_t;
                }
                
                // Take trace
                ComplexT tr = poly.trace() / Real(NCOLORS);
                re_sum += tr.real();
                im_sum += tr.imag();
            },
            poly_re, poly_im
        );
        
        // Average over spatial volume
        polyValue_ = ComplexT(poly_re / spatial_volume, 
                                  poly_im / spatial_volume);
        
        time_ = timer.seconds();
    }

    ComplexT value() const { return polyValue_; }
    Real absValue() const { return polyValue_.abs(); }
    double time() const { return time_; }

    /**
     * @brief Calculate number of floating point operations
     */
    long long flop() const {
        // Per spatial site: Nt matrix multiplications + 1 trace
        // Each NxN matrix multiply: ~8*N^3 flops
        int Nt = params_.grid[NDIMS - 1];
        int spatial_volume = 1;
        for (int i = 0; i < NDIMS - 1; ++i) {
            spatial_volume *= params_.grid[i];
        }
        long long flop_per_site = static_cast<long long>(Nt) * 8 * NCOLORS * NCOLORS * NCOLORS + 2 * NCOLORS;
        return flop_per_site * spatial_volume;
    }

    /**
     * @brief Calculate bytes read
     */
    long long bytes() const {
        // Read Nt temporal links per spatial site
        int Nt = params_.grid[NDIMS - 1];
        int spatial_volume = 1;
        for (int i = 0; i < NDIMS - 1; ++i) {
            spatial_volume *= params_.grid[i];
        }
        int nuparams_ = NCOLORS * NCOLORS * 2;
        return static_cast<long long>(Nt) * nuparams_ * sizeof(Real) * spatial_volume;
    }

    double flops() const {
        return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
    }

    double bandwidth() const {
        return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30)) : 0.0;
    }

    void printValue() const {
        printf("Polyakov Loop: %.12f + %.12f i (|P| = %.12f)\n",
               static_cast<double>(polyValue_.real()),
               static_cast<double>(polyValue_.imag()),
               static_cast<double>(absValue()));
    }

    void stat() const {
        printf("Polyakov Loop:  %.4f s\t%.2f GB/s\t%.2f GFlops\n",
               time_, bandwidth(), flops());
    }
};

//=============================================================================
// Reunitarization
//=============================================================================

/**
 * @brief Reunitarize gauge field to enforce SU(N) constraint
 * 
 * Uses Gram-Schmidt orthonormalization
 */
template<typename Real>
class Reunitarize {
public:
    using GaugeT = GaugeArray<Real>;
    using MatrixT = MatrixSun<Real, NCOLORS>;
    using ComplexT = Complex<Real>;

private:
    GaugeT& gauge_;
    LatticeParams params_;
    double time_;

    /**
     * @brief Gram-Schmidt reunitarization for a single matrix
     */
    KOKKOS_INLINE_FUNCTION
    static void reunitarizeMatrix(MatrixT& U) {
        if constexpr (NCOLORS == 3) {
            // For SU(3), use the simplified method
            // Normalize first row
            Real norm = Real(0);
            for (int j = 0; j < 3; ++j) {
                norm += U.e[0][j].abs2();
            }
            norm = Real(1) / Kokkos::sqrt(norm);
            for (int j = 0; j < 3; ++j) {
                U.e[0][j] *= norm;
            }
            
            // Orthogonalize second row to first
            ComplexT dot = ComplexT::zero();
            for (int j = 0; j < 3; ++j) {
                dot += ~U.e[0][j] * U.e[1][j];
            }
            for (int j = 0; j < 3; ++j) {
                U.e[1][j] -= dot * U.e[0][j];
            }
            
            // Normalize second row
            norm = Real(0);
            for (int j = 0; j < 3; ++j) {
                norm += U.e[1][j].abs2();
            }
            norm = Real(1) / Kokkos::sqrt(norm);
            for (int j = 0; j < 3; ++j) {
                U.e[1][j] *= norm;
            }
            
            // Third row is cross product of first two
            U.e[2][0] = ~(U.e[0][1] * U.e[1][2] - U.e[0][2] * U.e[1][1]);
            U.e[2][1] = ~(U.e[0][2] * U.e[1][0] - U.e[0][0] * U.e[1][2]);
            U.e[2][2] = ~(U.e[0][0] * U.e[1][1] - U.e[0][1] * U.e[1][0]);
        } else {
            // General Gram-Schmidt for SU(N)
            for (int row = 0; row < NCOLORS; ++row) {
                // Orthogonalize against previous rows
                for (int prev = 0; prev < row; ++prev) {
                    ComplexT dot = ComplexT::zero();
                    for (int j = 0; j < NCOLORS; ++j) {
                        dot += ~U.e[prev][j] * U.e[row][j];
                    }
                    for (int j = 0; j < NCOLORS; ++j) {
                        U.e[row][j] -= dot * U.e[prev][j];
                    }
                }
                
                // Normalize
                Real norm = Real(0);
                for (int j = 0; j < NCOLORS; ++j) {
                    norm += U.e[row][j].abs2();
                }
                norm = Real(1) / Kokkos::sqrt(norm);
                for (int j = 0; j < NCOLORS; ++j) {
                    U.e[row][j] *= norm;
                }
            }
        }
    }

public:
    Reunitarize(GaugeT& gauge, const LatticeParams& params)
        : gauge_(gauge), params_(params), time_(0) {}

    /**
     * @brief Reunitarize all links
     */
    void run() {
        Kokkos::Timer timer;
        
        auto gauge_view = gauge_.getView();
        int size = gauge_.size();  // volume * NDIMS
        int total_links = params_.size;  // volume * NDIMS
        
        Kokkos::parallel_for("Reunitarize",
            range_policy(0, total_links),
            KOKKOS_LAMBDA(const int link_idx) {
                ComplexT* gauge_ptr = gauge_view.data();
                
                // Load matrix from SOA format
                // Index: link_idx + elem_idx * size
                MatrixT U;
                for (int i = 0; i < NCOLORS; ++i) {
                    for (int j = 0; j < NCOLORS; ++j) {
                        U.e[i][j] = gauge_ptr[link_idx + 
                                   (j + i * NCOLORS) * size];
                    }
                }
                
                // Reunitarize
                reunitarizeMatrix(U);
                
                // Store back
                for (int i = 0; i < NCOLORS; ++i) {
                    for (int j = 0; j < NCOLORS; ++j) {
                        gauge_ptr[link_idx + 
                                 (j + i * NCOLORS) * size] = U.e[i][j];
                    }
                }
            }
        );
        Kokkos::fence();
        
        time_ = timer.seconds();
    }

    double time() const { return time_; }

    /**
     * @brief Calculate number of floating point operations
     */
    long long flop() const {
        // Gram-Schmidt per matrix: O(N^3) operations
        // For SU(3): ~100 flops per matrix
#if (NCOLORS == 3)
        long long flop_per_link = 100LL;
#else
        long long flop_per_link = static_cast<long long>(NCOLORS * NCOLORS * NCOLORS) * 3;
#endif
        return flop_per_link * params_.size;
    }

    /**
     * @brief Calculate bytes read/written
     */
    long long bytes() const {
        int nuparams_ = NCOLORS * NCOLORS * 2;
        // Read + write one matrix per link
        return 2LL * nuparams_ * sizeof(Real) * params_.size;
    }

    double flops() const {
        return (time_ > 0) ? (static_cast<double>(flop()) * 1.0e-9) / time_ : 0.0;
    }

    double bandwidth() const {
        return (time_ > 0) ? static_cast<double>(bytes()) / (time_ * (1LL << 30)) : 0.0;
    }

    void stat() const {
        printf("Reunitarize:  %.4f s\t%.2f GB/s\t%.2f GFlops\n",
               time_, bandwidth(), flops());
    }
};

} // namespace kwqft

#endif // KWQFT_MEASUREMENTS_HPP

