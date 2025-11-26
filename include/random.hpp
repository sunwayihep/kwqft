/**
 * @file random.hpp
 * @brief Random number generation for KWQFT using Kokkos
 * 
 * Provides portable random number generation using Kokkos Random pools
 */

#ifndef KWQFT_RANDOM_HPP
#define KWQFT_RANDOM_HPP

#include "kwqft_common.hpp"
#include "complex.hpp"
#include "matrixsun.hpp"
#include "msu2.hpp"

namespace kwqft {

/**
 * @brief Random number generator wrapper using Kokkos
 */
class RandomGenerator {
public:
    using PoolType = Kokkos::Random_XorShift64_Pool<DefaultExecSpace>;
    using generator_type = typename PoolType::generator_type;

private:
    PoolType pool_;
    int size_;
    unsigned int m_seed;
    bool m_initialized;

public:
    // Default constructor
    RandomGenerator() : size_(0), m_seed(0), m_initialized(false) {}

    // Constructor with seed
    RandomGenerator(unsigned int seed, int size) 
        : size_(size), m_seed(seed), m_initialized(false) {
        init(seed, size);
    }

    // Destructor
    ~RandomGenerator() = default;

    /**
     * @brief Initialize the random number pool
     */
    void init(unsigned int seed, int size = 0) {
        m_seed = seed;
        if (size > 0) size_ = size;
        
        // Initialize the random pool with the given seed
        pool_ = PoolType(seed);
        m_initialized = true;
    }

    /**
     * @brief Get the pool for use in parallel kernels
     */
    PoolType& getPool() { return pool_; }
    const PoolType& getPool() const { return pool_; }

    /**
     * @brief Check if initialized
     */
    bool initialized() const { return m_initialized; }

    /**
     * @brief Get seed
     */
    unsigned int seed() const { return m_seed; }

    /**
     * @brief Get size
     */
    int size() const { return size_; }

    /**
     * @brief Release resources
     */
    void release() {
        m_initialized = false;
    }
};

//=============================================================================
// Device-side random number functions
//=============================================================================

/**
 * @brief Generate uniform random number in [0, 1)
 */
template<typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION
Real Random(Generator& gen) {
    return gen.drand();
}

/**
 * @brief Generate uniform random number in [a, b)
 */
template<typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION
Real Random(Generator& gen, Real a, Real b) {
    return a + (b - a) * gen.drand();
}

/**
 * @brief Generate normal (Gaussian) random number
 */
template<typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION
Real RandomNormal(Generator& gen) {
    // Box-Muller transform
    Real u1 = gen.drand();
    Real u2 = gen.drand();
    // Avoid log(0)
    while (u1 < Real(1e-10)) u1 = gen.drand();
    return Kokkos::sqrt(Real(-2) * Kokkos::log(u1)) * Kokkos::cos(PII * u2);
}

/**
 * @brief Generate random SU(2) matrix using 4 real parameters
 */
template<typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION
Msu2<Real> randomSU2(Generator& gen) {
    Msu2<Real> a;
    Real aabs, ctheta, stheta, phi;
    
    a.a0() = Random<Real>(gen, Real(-1), Real(1));
    aabs = Kokkos::sqrt(Real(1) - a.a0() * a.a0());
    ctheta = Random<Real>(gen, Real(-1), Real(1));
    phi = PI * Random<Real>(gen);
    // Random sign for sin(theta)
    int sign = (gen.urand() & 1) ? 1 : -1;
    stheta = sign * Kokkos::sqrt(Real(1) - ctheta * ctheta);
    a.a1() = aabs * stheta * Kokkos::cos(phi);
    a.a2() = aabs * stheta * Kokkos::sin(phi);
    a.a3() = aabs * ctheta;
    
    return a;
}

/**
 * @brief Generate SU(2) matrix for heatbath using MILC algorithm
 */
template<typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION
Msu2<Real> generateSu2Matrix_milc(Real al, Generator& gen) {
    Real xr1, xr2, xr3, xr4, d, r;
    
    xr1 = gen.drand();
    xr1 = Kokkos::log(xr1 + Real(1e-10));
    xr2 = gen.drand();
    xr2 = Kokkos::log(xr2 + Real(1e-10));
    xr3 = gen.drand();
    xr4 = gen.drand();
    xr3 = Kokkos::cos(PI * xr3);
    d = -(xr2 + xr1 * xr3 * xr3) / al;
    
    int nacd = 0;
    if ((Real(1) - Real(0.5) * d) > xr4 * xr4) {
        nacd = 1;
    }
    
    Msu2<Real> a;
    
    if (nacd == 0 && al > Real(2)) {
        // Kennedy-Pendleton algorithm
        for (int k = 0; k < 20; ++k) {
            xr1 = gen.drand();
            xr1 = Kokkos::log(xr1 + Real(1e-10));
            xr2 = gen.drand();
            xr2 = Kokkos::log(xr2 + Real(1e-10));
            xr3 = gen.drand();
            xr4 = gen.drand();
            xr3 = Kokkos::cos(PI * xr3);
            d = -(xr2 + xr1 * xr3 * xr3) / al;
            if ((Real(1) - Real(0.5) * d) > xr4 * xr4) break;
        }
    }
    
    if (nacd == 0 && al <= Real(2)) {
        // Creutz algorithm
        xr3 = Kokkos::exp(Real(-2) * al);
        xr4 = Real(1) - xr3;
        for (int k = 0; k < 20; ++k) {
            xr1 = gen.drand();
            xr2 = gen.drand();
            r = xr3 + xr4 * xr1;
            a.a0() = Real(1) + Kokkos::log(r) / al;
            if ((Real(1) - a.a0() * a.a0()) > xr2 * xr2) break;
        }
        d = Real(1) - a.a0();
    }
    
    // Generate the four SU(2) elements
    a.a0() = Real(1) - d;
    xr3 = Real(1) - a.a0() * a.a0();
    if (xr3 < Real(0)) xr3 = Real(0);
    r = Kokkos::sqrt(xr3);
    
    a.a3() = (Real(2) * gen.drand() - Real(1)) * r;
    xr1 = xr3 - a.a3() * a.a3();
    if (xr1 < Real(0)) xr1 = Real(0);
    xr1 = Kokkos::sqrt(xr1);
    xr2 = PI * gen.drand();
    a.a1() = xr1 * Kokkos::cos(xr2);
    a.a2() = xr1 * Kokkos::sin(xr2);
    
    return a;
}

/**
 * @brief Generate random SU(N) matrix for hot start
 */
template<typename Real, typename Generator>
KOKKOS_INLINE_FUNCTION
MatrixSun<Real, NCOLORS> randomize(Generator& gen) {
    MatrixSun<Real, NCOLORS> U;
    
    for (int i = 0; i < NCOLORS; ++i) {
        for (int j = 0; j < NCOLORS; ++j) {
            U.e[i][j] = Complex<Real>(
                gen.drand() - Real(0.5),
                gen.drand() - Real(0.5)
            );
        }
    }
    
    return U;
}

// Type alias for convenience
using RNG = RandomGenerator;

} // namespace kwqft

#endif // KWQFT_RANDOM_HPP

