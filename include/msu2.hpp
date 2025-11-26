/**
 * @file msu2.hpp
 * @brief SU(2) subgroup representation for pseudo-heatbath algorithm
 * 
 * Uses 4 real numbers (a0, a1, a2, a3) to represent an SU(2) matrix:
 * U = a0*I + i*(a1*sigma1 + a2*sigma2 + a3*sigma3)
 * where a0^2 + a1^2 + a2^2 + a3^2 = 1
 */

#ifndef KWQFT_Msu2_HPP
#define KWQFT_Msu2_HPP

#include "kwqft_common.hpp"
#include "complex.hpp"

namespace kwqft {

/**
 * @brief SU(2) matrix represented by 4 real parameters
 * @tparam Real The underlying real type (float or double)
 */
template<typename Real>
class Msu2 {
public:
    Real m_a[4];  // a0, a1, a2, a3

    // Constructors
    KOKKOS_INLINE_FUNCTION
    Msu2() {
        m_a[0] = Real(1);
        m_a[1] = Real(0);
        m_a[2] = Real(0);
        m_a[3] = Real(0);
    }

    KOKKOS_INLINE_FUNCTION
    Msu2(Real a0, Real a1, Real a2, Real a3) {
        m_a[0] = a0;
        m_a[1] = a1;
        m_a[2] = a2;
        m_a[3] = a3;
    }

    KOKKOS_INLINE_FUNCTION
    Msu2(const Msu2& other) {
        m_a[0] = other.m_a[0];
        m_a[1] = other.m_a[1];
        m_a[2] = other.m_a[2];
        m_a[3] = other.m_a[3];
    }

    // Accessors
    KOKKOS_INLINE_FUNCTION Real& a0() { return m_a[0]; }
    KOKKOS_INLINE_FUNCTION Real& a1() { return m_a[1]; }
    KOKKOS_INLINE_FUNCTION Real& a2() { return m_a[2]; }
    KOKKOS_INLINE_FUNCTION Real& a3() { return m_a[3]; }

    KOKKOS_INLINE_FUNCTION Real a0() const { return m_a[0]; }
    KOKKOS_INLINE_FUNCTION Real a1() const { return m_a[1]; }
    KOKKOS_INLINE_FUNCTION Real a2() const { return m_a[2]; }
    KOKKOS_INLINE_FUNCTION Real a3() const { return m_a[3]; }

    // Assignment
    KOKKOS_INLINE_FUNCTION
    Msu2& operator=(const Msu2& other) {
        m_a[0] = other.m_a[0];
        m_a[1] = other.m_a[1];
        m_a[2] = other.m_a[2];
        m_a[3] = other.m_a[3];
        return *this;
    }

    // Absolute value (should be 1 for normalized SU(2))
    KOKKOS_INLINE_FUNCTION
    Real abs() const {
        return Kokkos::sqrt(m_a[0]*m_a[0] + m_a[1]*m_a[1] + 
                           m_a[2]*m_a[2] + m_a[3]*m_a[3]);
    }

    // Squared norm
    KOKKOS_INLINE_FUNCTION
    Real abs2() const {
        return m_a[0]*m_a[0] + m_a[1]*m_a[1] + m_a[2]*m_a[2] + m_a[3]*m_a[3];
    }

    // Normalize
    KOKKOS_INLINE_FUNCTION
    Msu2 normalize() const {
        Real norm = Real(1) / abs();
        return Msu2(m_a[0]*norm, m_a[1]*norm, m_a[2]*norm, m_a[3]*norm);
    }

    // Conjugate (for SU(2): a0 stays same, others flip sign)
    KOKKOS_INLINE_FUNCTION
    Msu2 conj() const {
        return Msu2(m_a[0], -m_a[1], -m_a[2], -m_a[3]);
    }

    // Conjugate and normalize
    KOKKOS_INLINE_FUNCTION
    Msu2 conj_normalize() const {
        Real norm = Real(1) / abs();
        return Msu2(m_a[0]*norm, -m_a[1]*norm, -m_a[2]*norm, -m_a[3]*norm);
    }

    // Multiplication by scalar
    KOKKOS_INLINE_FUNCTION
    Msu2& operator*=(Real s) {
        m_a[0] *= s;
        m_a[1] *= s;
        m_a[2] *= s;
        m_a[3] *= s;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Msu2 operator*(Real s) const {
        return Msu2(m_a[0]*s, m_a[1]*s, m_a[2]*s, m_a[3]*s);
    }

    // Print
    void print() const {
        printf("Msu2: (%.6f, %.6f, %.6f, %.6f)\n",
               static_cast<double>(m_a[0]), static_cast<double>(m_a[1]),
               static_cast<double>(m_a[2]), static_cast<double>(m_a[3]));
    }
};

/**
 * @brief Multiply two SU(2) matrices: result = U * V
 * Using quaternion multiplication
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
Msu2<Real> mulsu2(const Msu2<Real>& u, const Msu2<Real>& v) {
    return Msu2<Real>(
        u.a0()*v.a0() - u.a1()*v.a1() - u.a2()*v.a2() - u.a3()*v.a3(),
        u.a0()*v.a1() + u.a1()*v.a0() - u.a2()*v.a3() + u.a3()*v.a2(),
        u.a0()*v.a2() + u.a1()*v.a3() + u.a2()*v.a0() - u.a3()*v.a1(),
        u.a0()*v.a3() - u.a1()*v.a2() + u.a2()*v.a1() + u.a3()*v.a0()
    );
}

/**
 * @brief Multiply U * V^\dagger for SU(2) matrices
 */
template<typename Real>
KOKKOS_INLINE_FUNCTION
Msu2<Real> mulsu2UVDagger(const Msu2<Real>& u, const Msu2<Real>& v) {
    return Msu2<Real>(
        u.a0()*v.a0() + u.a1()*v.a1() + u.a2()*v.a2() + u.a3()*v.a3(),
        -u.a0()*v.a1() + u.a1()*v.a0() + u.a2()*v.a3() - u.a3()*v.a2(),
        -u.a0()*v.a2() - u.a1()*v.a3() + u.a2()*v.a0() + u.a3()*v.a1(),
        -u.a0()*v.a3() + u.a1()*v.a2() - u.a2()*v.a1() + u.a3()*v.a0()
    );
}

// Type alias
template<typename Real>
using msu2 = Msu2<Real>;

} // namespace kwqft

#endif // KWQFT_Msu2_HPP

