/**
 * @file complex.hpp
 * @brief Kokkos-portable complex number class for KWQFT
 * 
 * Provides a complex number implementation that works on all Kokkos backends
 */

#ifndef KWQFT_COMPLEX_HPP
#define KWQFT_COMPLEX_HPP

#include "kwqft_common.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

namespace kwqft {

/**
 * @brief Complex number class compatible with Kokkos
 * @tparam Real The underlying real type (float or double)
 */
template<typename Real>
class Complex {
public:
    Real x;  // Real part
    Real y;  // Imaginary part

    // Constructors
    KOKKOS_INLINE_FUNCTION
    Complex() : x(Real(0)), y(Real(0)) {}

    KOKKOS_INLINE_FUNCTION
    Complex(Real re) : x(re), y(Real(0)) {}

    KOKKOS_INLINE_FUNCTION
    Complex(Real re, Real im) : x(re), y(im) {}

    KOKKOS_INLINE_FUNCTION
    Complex(const Complex& other) : x(other.x), y(other.y) {}

    // Convert from Kokkos::complex if needed
    KOKKOS_INLINE_FUNCTION
    Complex(const Kokkos::complex<Real>& kc) : x(kc.real()), y(kc.imag()) {}

    // Assignment operators
    KOKKOS_INLINE_FUNCTION
    Complex& operator=(const Complex& other) {
        x = other.x;
        y = other.y;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator=(Real re) {
        x = re;
        y = Real(0);
        return *this;
    }

    // Accessors
    KOKKOS_INLINE_FUNCTION Real& real() { return x; }
    KOKKOS_INLINE_FUNCTION Real& imag() { return y; }
    KOKKOS_INLINE_FUNCTION Real real() const { return x; }
    KOKKOS_INLINE_FUNCTION Real imag() const { return y; }

    // Comparison operators
    KOKKOS_INLINE_FUNCTION
    bool operator==(const Complex& other) const {
        return (x == other.x) && (y == other.y);
    }

    KOKKOS_INLINE_FUNCTION
    bool operator!=(const Complex& other) const {
        return !(*this == other);
    }

    // Arithmetic operators: Addition
    KOKKOS_INLINE_FUNCTION
    Complex operator+(const Complex& b) const {
        return Complex(x + b.x, y + b.y);
    }

    KOKKOS_INLINE_FUNCTION
    Complex operator+(Real b) const {
        return Complex(x + b, y);
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator+=(const Complex& b) {
        x += b.x;
        y += b.y;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator+=(Real b) {
        x += b;
        return *this;
    }

    // Arithmetic operators: Subtraction
    KOKKOS_INLINE_FUNCTION
    Complex operator-(const Complex& b) const {
        return Complex(x - b.x, y - b.y);
    }

    KOKKOS_INLINE_FUNCTION
    Complex operator-(Real b) const {
        return Complex(x - b, y);
    }

    KOKKOS_INLINE_FUNCTION
    Complex operator-() const {
        return Complex(-x, -y);
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator-=(const Complex& b) {
        x -= b.x;
        y -= b.y;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator-=(Real b) {
        x -= b;
        return *this;
    }

    // Arithmetic operators: Multiplication
    KOKKOS_INLINE_FUNCTION
    Complex operator*(const Complex& b) const {
        return Complex(x * b.x - y * b.y, y * b.x + x * b.y);
    }

    KOKKOS_INLINE_FUNCTION
    Complex operator*(Real b) const {
        return Complex(x * b, y * b);
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator*=(const Complex& b) {
        Real tmp = x * b.x - y * b.y;
        y = y * b.x + x * b.y;
        x = tmp;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator*=(Real b) {
        x *= b;
        y *= b;
        return *this;
    }

    // Arithmetic operators: Division
    KOKKOS_INLINE_FUNCTION
    Complex operator/(const Complex& b) const {
        Real denom = Real(1) / (b.x * b.x + b.y * b.y);
        return Complex((x * b.x + y * b.y) * denom,
                       (y * b.x - x * b.y) * denom);
    }

    KOKKOS_INLINE_FUNCTION
    Complex operator/(Real b) const {
        return Complex(x / b, y / b);
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator/=(const Complex& b) {
        Real denom = Real(1) / (b.x * b.x + b.y * b.y);
        Real tmp = (x * b.x + y * b.y) * denom;
        y = (y * b.x - x * b.y) * denom;
        x = tmp;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    Complex& operator/=(Real b) {
        x /= b;
        y /= b;
        return *this;
    }

    // Complex conjugate (using ~ operator like original code)
    KOKKOS_INLINE_FUNCTION
    Complex operator~() const {
        return Complex(x, -y);
    }

    KOKKOS_INLINE_FUNCTION
    Complex conj() const {
        return Complex(x, -y);
    }

    // Absolute value (modulus)
    KOKKOS_INLINE_FUNCTION
    Real abs() const {
        return Kokkos::sqrt(x * x + y * y);
    }

    // Squared modulus
    KOKKOS_INLINE_FUNCTION
    Real abs2() const {
        return x * x + y * y;
    }

    // Phase angle
    KOKKOS_INLINE_FUNCTION
    Real phase() const {
        return Kokkos::atan2(y, x);
    }

    KOKKOS_INLINE_FUNCTION
    Real arg() const {
        return Kokkos::atan2(y, x);
    }

    // Static factory methods
    KOKKOS_INLINE_FUNCTION
    static Complex make_complex(Real re, Real im) {
        return Complex(re, im);
    }

    KOKKOS_INLINE_FUNCTION
    static Complex zero() {
        return Complex(Real(0), Real(0));
    }

    KOKKOS_INLINE_FUNCTION
    static Complex one() {
        return Complex(Real(1), Real(0));
    }

    KOKKOS_INLINE_FUNCTION
    static Complex unit() {
        return Complex(Real(1), Real(0));
    }

    KOKKOS_INLINE_FUNCTION
    static Complex I() {
        return Complex(Real(0), Real(1));
    }

    // Print (only works on host)
    void print() const {
        printf("%.10e + %.10ej\n", static_cast<double>(x), static_cast<double>(y));
    }
};

// Friend operators for Real + Complex, Real * Complex, etc.
template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> operator+(Real a, const Complex<Real>& z) {
    return Complex<Real>(z.x + a, z.y);
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> operator-(Real a, const Complex<Real>& z) {
    return Complex<Real>(a - z.x, -z.y);
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> operator*(Real a, const Complex<Real>& z) {
    return Complex<Real>(z.x * a, z.y * a);
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> operator/(Real a, const Complex<Real>& b) {
    Real denom = Real(1) / (b.x * b.x + b.y * b.y);
    return Complex<Real>(a * b.x * denom, -a * b.y * denom);
}

// Math functions
template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> conj(const Complex<Real>& z) {
    return Complex<Real>(z.x, -z.y);
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Real abs(const Complex<Real>& z) {
    return Kokkos::sqrt(z.x * z.x + z.y * z.y);
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Real abs2(const Complex<Real>& z) {
    return z.x * z.x + z.y * z.y;
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> sqrt(const Complex<Real>& z) {
    if (z.x == Real(0) && z.y == Real(0)) {
        return Complex<Real>(Real(0), Real(0));
    }
    Real r = z.abs();
    Real theta = z.arg() / Real(2);
    Real sqrtr = Kokkos::sqrt(r);
    return Complex<Real>(sqrtr * Kokkos::cos(theta), sqrtr * Kokkos::sin(theta));
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> exp(const Complex<Real>& z) {
    Real rho = Kokkos::exp(z.x);
    return Complex<Real>(rho * Kokkos::cos(z.y), rho * Kokkos::sin(z.y));
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> log(const Complex<Real>& z) {
    return Complex<Real>(Kokkos::log(z.abs()), z.arg());
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> polar(Real rho, Real theta) {
    return Complex<Real>(rho * Kokkos::cos(theta), rho * Kokkos::sin(theta));
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> sin(const Complex<Real>& z) {
    return Complex<Real>(Kokkos::sin(z.x) * Kokkos::cosh(z.y),
                         Kokkos::cos(z.x) * Kokkos::sinh(z.y));
}

template<typename Real>
KOKKOS_INLINE_FUNCTION
Complex<Real> cos(const Complex<Real>& z) {
    return Complex<Real>(Kokkos::cos(z.x) * Kokkos::cosh(z.y),
                        -Kokkos::sin(z.x) * Kokkos::sinh(z.y));
}

// Type aliases
using complexs = Complex<float>;
using complexd = Complex<double>;

// Template alias for generic use
template<typename Real>
using complex = Complex<Real>;

} // namespace kwqft

#endif // KWQFT_COMPLEX_HPP

