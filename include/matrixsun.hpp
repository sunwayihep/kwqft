/**
 * @file matrixsun.hpp
 * @brief SU(N) matrix class for Kokkos-portable lattice QCD
 * 
 * Provides a templated SU(N) matrix implementation that works
 * efficiently on all Kokkos backends
 */

#ifndef KWQFT_MATRIXSUN_HPP
#define KWQFT_MATRIXSUN_HPP

#include "kwqft_common.hpp"
#include "complex.hpp"
#include "msu2.hpp"

namespace kwqft {

/**
 * @brief SU(N) matrix class
 * @tparam Real The underlying real type (float or double)
 * @tparam Nc Number of colors
 */
template<typename Real, int Nc = NCOLORS>
class MatrixSun {
public:
    Complex<Real> e[Nc][Nc];  // Matrix elements

    // Default constructor (uninitialized for performance)
    KOKKOS_INLINE_FUNCTION
    MatrixSun() {}

    // Copy constructor
    KOKKOS_INLINE_FUNCTION
    MatrixSun(const MatrixSun& other) {
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                e[i][j] = other.e[i][j];
            }
        }
    }

    // Element access
    KOKKOS_INLINE_FUNCTION
    Complex<Real>& operator()(int i, int j) {
        return e[i][j];
    }

    KOKKOS_INLINE_FUNCTION
    Complex<Real> operator()(int i, int j) const {
        return e[i][j];
    }

    // Assignment
    KOKKOS_INLINE_FUNCTION
    MatrixSun& operator=(const MatrixSun& other) {
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                e[i][j] = other.e[i][j];
            }
        }
        return *this;
    }

    //=========================================================================
    // Addition operations
    //=========================================================================
    KOKKOS_INLINE_FUNCTION
    MatrixSun operator+(const MatrixSun& A) const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = e[i][j] + A.e[i][j];
            }
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    MatrixSun& operator+=(const MatrixSun& A) {
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                e[i][j] += A.e[i][j];
            }
        }
        return *this;
    }

    //=========================================================================
    // Subtraction operations
    //=========================================================================
    KOKKOS_INLINE_FUNCTION
    MatrixSun operator-(const MatrixSun& A) const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = e[i][j] - A.e[i][j];
            }
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    MatrixSun& operator-=(const MatrixSun& A) {
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                e[i][j] -= A.e[i][j];
            }
        }
        return *this;
    }

    // Negation
    KOKKOS_INLINE_FUNCTION
    MatrixSun operator-() const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = -e[i][j];
            }
        }
        return res;
    }

    //=========================================================================
    // Multiplication operations
    //=========================================================================
    
    // Matrix-matrix multiplication
    KOKKOS_INLINE_FUNCTION
    MatrixSun operator*(const MatrixSun& A) const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = e[i][0] * A.e[0][j];
                for (int k = 1; k < Nc; ++k) {
                    res.e[i][j] += e[i][k] * A.e[k][j];
                }
            }
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    MatrixSun& operator*=(const MatrixSun& A) {
        *this = (*this) * A;
        return *this;
    }

    // Scalar multiplication
    KOKKOS_INLINE_FUNCTION
    MatrixSun operator*(Real s) const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = e[i][j] * s;
            }
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    MatrixSun& operator*=(Real s) {
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                e[i][j] *= s;
            }
        }
        return *this;
    }

    // Complex scalar multiplication
    KOKKOS_INLINE_FUNCTION
    MatrixSun operator*(const Complex<Real>& c) const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = e[i][j] * c;
            }
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    MatrixSun& operator*=(const Complex<Real>& c) {
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                e[i][j] *= c;
            }
        }
        return *this;
    }

    //=========================================================================
    // Division operations
    //=========================================================================
    KOKKOS_INLINE_FUNCTION
    MatrixSun operator/(Real s) const {
        MatrixSun res;
        Real inv = Real(1) / s;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = e[i][j] * inv;
            }
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    MatrixSun& operator/=(Real s) {
        Real inv = Real(1) / s;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                e[i][j] *= inv;
            }
        }
        return *this;
    }

    //=========================================================================
    // Matrix operations
    //=========================================================================

    // Hermitian conjugate (dagger)
    KOKKOS_INLINE_FUNCTION
    MatrixSun dagger() const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = ~e[j][i];  // Transpose and conjugate
            }
        }
        return res;
    }

    // Complex conjugate only (no transpose)
    KOKKOS_INLINE_FUNCTION
    MatrixSun conj() const {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = ~e[i][j];
            }
        }
        return res;
    }

    // Trace
    KOKKOS_INLINE_FUNCTION
    Complex<Real> trace() const {
        Complex<Real> tr = Complex<Real>::zero();
        for (int i = 0; i < Nc; ++i) {
            tr += e[i][i];
        }
        return tr;
    }

    // Real part of trace
    KOKKOS_INLINE_FUNCTION
    Real realtrace() const {
        Real tr = Real(0);
        for (int i = 0; i < Nc; ++i) {
            tr += e[i][i].real();
        }
        return tr;
    }

    // Determinant (for SU(3) using specific formula, general for others)
    KOKKOS_INLINE_FUNCTION
    Complex<Real> det() const {
        if constexpr (Nc == 3) {
            Complex<Real> res;
            res = e[0][1] * e[1][2] * e[2][0];
            res -= e[0][2] * e[1][1] * e[2][0];
            res += e[0][2] * e[1][0] * e[2][1];
            res -= e[0][0] * e[1][2] * e[2][1];
            res -= e[0][1] * e[1][0] * e[2][2];
            res += e[0][0] * e[1][1] * e[2][2];
            return res;
        } else if constexpr (Nc == 2) {
            return e[0][0] * e[1][1] - e[0][1] * e[1][0];
        } else {
            // General LU decomposition based determinant
            MatrixSun<Real, Nc> b;
            for (int i = 0; i < Nc; ++i)
                for (int j = 0; j < Nc; ++j)
                    b.e[i][j] = e[i][j];
            
            Complex<Real> res;
            for (int j = 0; j < Nc; ++j) {
                for (int i = 0; i <= j; ++i) {
                    res = b.e[j][i];
                    for (int c = 0; c < i; ++c)
                        res -= b.e[c][i] * b.e[j][c];
                    b.e[j][i] = res;
                }
                for (int i = (j + 1); i < Nc; ++i) {
                    res = b.e[j][i];
                    for (int c = 0; c < j; ++c)
                        res -= b.e[c][i] * b.e[j][c];
                    b.e[j][i] = b.e[j][j].conj() * res / b.e[j][j].abs2();
                }
            }
            res = b.e[0][0] * b.e[1][1];
            for (int c = 2; c < Nc; ++c)
                res *= b.e[c][c];
            return res;
        }
    }

    // Subtract trace * identity / Nc (make traceless)
    KOKKOS_INLINE_FUNCTION
    MatrixSun subtraceunit() const {
        Complex<Real> tr = trace() / Real(Nc);
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = (i == j) ? e[i][j] - tr : e[i][j];
            }
        }
        return res;
    }

    //=========================================================================
    // Static factory methods
    //=========================================================================

    // Zero matrix
    KOKKOS_INLINE_FUNCTION
    static MatrixSun zero() {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = Complex<Real>::zero();
            }
        }
        return res;
    }

    // Identity matrix
    KOKKOS_INLINE_FUNCTION
    static MatrixSun identity() {
        MatrixSun res;
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                res.e[i][j] = (i == j) ? Complex<Real>::one() : Complex<Real>::zero();
            }
        }
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    static MatrixSun unit() {
        return identity();
    }

    //=========================================================================
    // Print (host only)
    //=========================================================================
    void print() const {
        for (int i = 0; i < Nc; ++i) {
            for (int j = 0; j < Nc; ++j) {
                if (i == 0 && j == 0) printf("[ ");
                else printf("  ");
                printf("%.10e + %.10ej", 
                       static_cast<double>(e[i][j].real()),
                       static_cast<double>(e[i][j].imag()));
                if (i == Nc-1 && j == Nc-1) printf(" ]\n");
                else printf("\t");
            }
            printf("\n");
        }
    }
};

//=============================================================================
// Free functions for matrix operations
//=============================================================================

/**
 * @brief Compute A^\dagger * B
 */
template<typename Real, int Nc>
KOKKOS_INLINE_FUNCTION
MatrixSun<Real, Nc> UDaggerU(const MatrixSun<Real, Nc>& A, 
                             const MatrixSun<Real, Nc>& B) {
    MatrixSun<Real, Nc> C;
    for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
            C.e[i][j] = ~A.e[0][i] * B.e[0][j];
            for (int k = 1; k < Nc; ++k) {
                C.e[i][j] += ~A.e[k][i] * B.e[k][j];
            }
        }
    }
    return C;
}

/**
 * @brief Compute A * B^\dagger
 */
template<typename Real, int Nc>
KOKKOS_INLINE_FUNCTION
MatrixSun<Real, Nc> UUDagger(const MatrixSun<Real, Nc>& A,
                             const MatrixSun<Real, Nc>& B) {
    MatrixSun<Real, Nc> C;
    for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
            C.e[i][j] = A.e[i][0] * ~B.e[j][0];
            for (int k = 1; k < Nc; ++k) {
                C.e[i][j] += A.e[i][k] * ~B.e[j][k];
            }
        }
    }
    return C;
}

/**
 * @brief Real part of trace(A^\dagger * B)
 */
template<typename Real, int Nc>
KOKKOS_INLINE_FUNCTION
Real UDaggerURealTrace(const MatrixSun<Real, Nc>& A,
                       const MatrixSun<Real, Nc>& B) {
    Real res = Real(0);
    for (int i = 0; i < Nc; ++i) {
        for (int k = 0; k < Nc; ++k) {
            res += A.e[k][i].real() * B.e[k][i].real() +
                   A.e[k][i].imag() * B.e[k][i].imag();
        }
    }
    return res;
}

/**
 * @brief Real part of trace(A * B)
 */
template<typename Real, int Nc>
KOKKOS_INLINE_FUNCTION
Real realtraceUV(const MatrixSun<Real, Nc>& a, const MatrixSun<Real, Nc>& b) {
    Real sum = Real(0);
    for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
            sum += a.e[i][j].real() * b.e[j][i].real() -
                   a.e[i][j].imag() * b.e[j][i].imag();
        }
    }
    return sum;
}

/**
 * @brief Real part of trace(A * B^\dagger)
 */
template<typename Real, int Nc>
KOKKOS_INLINE_FUNCTION
Real realtraceUVdagger(const MatrixSun<Real, Nc>& a, const MatrixSun<Real, Nc>& b) {
    Real sum = Real(0);
    for (int i = 0; i < Nc; ++i) {
        for (int j = 0; j < Nc; ++j) {
            sum += a.e[i][j].real() * b.e[i][j].real() +
                   a.e[i][j].imag() * b.e[i][j].imag();
        }
    }
    return sum;
}

//=============================================================================
// SU(2) subgroup operations
//=============================================================================

/**
 * @brief Calculate the SU(2) index block indices
 */
KOKKOS_INLINE_FUNCTION
void IndexBlock(int block, int& p, int& q) {
    if constexpr (NCOLORS == 3) {
        if (block == 0) { p = 0; q = 1; }
        else if (block == 1) { p = 1; q = 2; }
        else { p = 0; q = 2; }
    } else {
        int i1;
        int found = 0;
        int del_i = 0;
        int index = -1;
        while (del_i < (NCOLORS - 1) && found == 0) {
            del_i++;
            for (i1 = 0; i1 < (NCOLORS - del_i); i1++) {
                index++;
                if (index == block) {
                    found = 1;
                    break;
                }
            }
        }
        q = i1 + del_i;
        p = i1;
    }
}

/**
 * @brief Extract SU(2) subgroup from SU(N) matrix
 */
template<typename Real, int Nc>
KOKKOS_INLINE_FUNCTION
Msu2<Real> getBlockSu2(const MatrixSun<Real, Nc>& tmp1, int p, int q) {
    Msu2<Real> r;
    r.a0() = tmp1.e[p][p].real() + tmp1.e[q][q].real();
    r.a1() = tmp1.e[p][q].imag() + tmp1.e[q][p].imag();
    r.a2() = tmp1.e[p][q].real() - tmp1.e[q][p].real();
    r.a3() = tmp1.e[p][p].imag() - tmp1.e[q][q].imag();
    return r;
}

/**
 * @brief Multiply SU(N) matrix by SU(2) subgroup: link <- u * link
 */
template<typename Real, int Nc>
KOKKOS_INLINE_FUNCTION
void mulBlockSun(Msu2<Real> u, MatrixSun<Real, Nc>& link, int p, int q) {
    Complex<Real> tmp;
    Complex<Real> a00(u.a0(), u.a3());
    Complex<Real> a01(u.a2(), u.a1());
    Complex<Real> a10(-u.a2(), u.a1());
    Complex<Real> a11(u.a0(), -u.a3());
    
    for (int j = 0; j < Nc; ++j) {
        tmp = a00 * link.e[p][j] + a01 * link.e[q][j];
        link.e[q][j] = a10 * link.e[p][j] + a11 * link.e[q][j];
        link.e[p][j] = tmp;
    }
}

//=============================================================================
// Type aliases
//=============================================================================

template<typename Real>
using msun = MatrixSun<Real, NCOLORS>;

template<typename Real>
using msu3 = MatrixSun<Real, NCOLORS>;  // For compatibility

using msuns = MatrixSun<float, NCOLORS>;
using msund = MatrixSun<double, NCOLORS>;

} // namespace kwqft

#endif // KWQFT_MATRIXSUN_HPP

