/**
 * @file matrixsun.hpp
 * @brief SU(N) matrix class for Kokkos-portable lattice QCD
 *
 * Provides a templated SU(N) matrix implementation that works
 * efficiently on all Kokkos backends
 */

#ifndef KWQFT_MATRIXSUN_HPP
#define KWQFT_MATRIXSUN_HPP

#include "complex.hpp"
#include "kwqft_common.hpp"
#include "msu2.hpp"

namespace kwqft {

/// SIMD-pack GEMM. Kokkos SIMD operators map to separate multiply and add
/// instructions, so the complex products are written as explicit FMAs. Terms
/// that enter with a minus sign go to separate accumulators (Kokkos has no
/// fused multiply-subtract), which also gives four independent FMA chains.
template <bool HermA, bool HermB, typename Real, int Nc>
KOKKOS_INLINE_FUNCTION void gemmIjkFma(const Complex<Real> (&a)[Nc][Nc],
                                         const Complex<Real> (&b)[Nc][Nc],
                                         Complex<Real> (&c)[Nc][Nc]) {
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      Real xp(0), xm(0), yp(0), ym(0);
      for (int k = 0; k < Nc; ++k) {
        const Complex<Real> &ak = HermA ? a[k][i] : a[i][k];
        const Complex<Real> &bk = HermB ? b[j][k] : b[k][j];
        xp = Kokkos::fma(ak.x, bk.x, xp);
        if constexpr (HermA != HermB) {
          xp = Kokkos::fma(ak.y, bk.y, xp);
        } else {
          xm = Kokkos::fma(ak.y, bk.y, xm);
        }
        if constexpr (HermB) {
          ym = Kokkos::fma(ak.x, bk.y, ym);
        } else {
          yp = Kokkos::fma(ak.x, bk.y, yp);
        }
        if constexpr (HermA) {
          ym = Kokkos::fma(ak.y, bk.x, ym);
        } else {
          yp = Kokkos::fma(ak.y, bk.x, yp);
        }
      }
      c[i][j] = Complex<Real>(xp - xm, yp - ym);
    }
  }
}

template <bool HermA, bool HermB, typename Real, int Nc>
KWQFT_INLINE_FUNCTION void gemmIjk(const Complex<Real> (&a)[Nc][Nc],
                                    const Complex<Real> (&b)[Nc][Nc],
                                    Complex<Real> (&c)[Nc][Nc]) {
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      Complex<Real> s;
      if constexpr (HermA && HermB) {
        s = ~a[0][i] * ~b[j][0];
      } else if constexpr (HermA) {
        s = ~a[0][i] * b[0][j];
      } else if constexpr (HermB) {
        s = a[i][0] * ~b[j][0];
      } else {
        s = a[i][0] * b[0][j];
      }
      for (int k = 1; k < Nc; ++k) {
        if constexpr (HermA && HermB) {
          s += ~a[k][i] * ~b[j][k];
        } else if constexpr (HermA) {
          s += ~a[k][i] * b[k][j];
        } else if constexpr (HermB) {
          s += a[i][k] * ~b[j][k];
        } else {
          s += a[i][k] * b[k][j];
        }
      }
      c[i][j] = s;
    }
  }
}

/// C = op(A) * op(B). HermX selects conjugate-transpose.
template <bool HermA, bool HermB, typename Real, int Nc>
KWQFT_INLINE_FUNCTION void sunGemm(const Complex<Real> (&a)[Nc][Nc],
                                    const Complex<Real> (&b)[Nc][Nc],
                                    Complex<Real> (&c)[Nc][Nc]) {
  if constexpr (std::is_floating_point_v<Real>) {
    gemmIjk<HermA, HermB>(a, b, c);
  } else {
    gemmIjkFma<HermA, HermB>(a, b, c);
  }
}

/**
 * @brief SU(N) matrix class
 * @tparam Real The underlying real type (float or double)
 * @tparam Nc Number of colors
 */
template <typename Real, int Nc = NCOLORS> class MatrixSun {
public:
  Complex<Real> e[Nc][Nc]; // Matrix elements

  // Default constructor (uninitialized for performance). Copy is implicit so
  // the Nc x Nc block stays trivially copyable.
  KOKKOS_INLINE_FUNCTION
  MatrixSun() {}

  // Element access
  KOKKOS_INLINE_FUNCTION
  Complex<Real> &operator()(int i, int j) { return e[i][j]; }

  KOKKOS_INLINE_FUNCTION
  Complex<Real> operator()(int i, int j) const { return e[i][j]; }

  //=========================================================================
  // Addition operations
  //=========================================================================
  KOKKOS_INLINE_FUNCTION
  MatrixSun operator+(const MatrixSun &a) const {
    MatrixSun res;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        res.e[i][j] = e[i][j] + a.e[i][j];
      }
    }
    return res;
  }

  KOKKOS_INLINE_FUNCTION
  MatrixSun &operator+=(const MatrixSun &a) {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        e[i][j] += a.e[i][j];
      }
    }
    return *this;
  }

  //=========================================================================
  // Subtraction operations
  //=========================================================================
  KOKKOS_INLINE_FUNCTION
  MatrixSun operator-(const MatrixSun &a) const {
    MatrixSun res;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        res.e[i][j] = e[i][j] - a.e[i][j];
      }
    }
    return res;
  }

  KOKKOS_INLINE_FUNCTION
  MatrixSun &operator-=(const MatrixSun &a) {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        e[i][j] -= a.e[i][j];
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
  KWQFT_INLINE_FUNCTION
  MatrixSun operator*(const MatrixSun &a) const {
    MatrixSun res;
    sunGemm<false, false>(e, a.e, res.e);
    return res;
  }

  KOKKOS_INLINE_FUNCTION
  MatrixSun &operator*=(const MatrixSun &a) {
    *this = (*this) * a;
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
  MatrixSun &operator*=(Real s) {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        e[i][j] *= s;
      }
    }
    return *this;
  }

  // Complex scalar multiplication
  KOKKOS_INLINE_FUNCTION
  MatrixSun operator*(const Complex<Real> &c) const {
    MatrixSun res;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        res.e[i][j] = e[i][j] * c;
      }
    }
    return res;
  }

  KOKKOS_INLINE_FUNCTION
  MatrixSun &operator*=(const Complex<Real> &c) {
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
  MatrixSun &operator/=(Real s) {
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
        res.e[i][j] = ~e[j][i]; // Transpose and conjugate
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
  static MatrixSun unit() { return identity(); }

  //=========================================================================
  // Print (host only)
  //=========================================================================
  void print() const {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        if (i == 0 && j == 0)
          printf("[ ");
        else
          printf("  ");
        printf("%.10e + %.10ej", static_cast<double>(e[i][j].real()),
               static_cast<double>(e[i][j].imag()));
        if (i == Nc - 1 && j == Nc - 1)
          printf(" ]\n");
        else
          printf("\t");
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
template <typename Real, int Nc>
KOKKOS_INLINE_FUNCTION MatrixSun<Real, Nc>
uDaggerU(const MatrixSun<Real, Nc> &a, const MatrixSun<Real, Nc> &b) {
  MatrixSun<Real, Nc> c;
  sunGemm<true, false>(a.e, b.e, c.e);
  return c;
}

/**
 * @brief Compute A * B^\dagger
 */
template <typename Real, int Nc>
KWQFT_INLINE_FUNCTION MatrixSun<Real, Nc>
uuDagger(const MatrixSun<Real, Nc> &a, const MatrixSun<Real, Nc> &b) {
  MatrixSun<Real, Nc> c;
  sunGemm<false, true>(a.e, b.e, c.e);
  return c;
}

/**
 * @brief Real part of trace(A^\dagger * B)
 */
template <typename Real, int Nc>
KOKKOS_INLINE_FUNCTION Real uDaggerURealTrace(const MatrixSun<Real, Nc> &a,
                                              const MatrixSun<Real, Nc> &b) {
  Real res = Real(0);
  for (int i = 0; i < Nc; ++i) {
    for (int k = 0; k < Nc; ++k) {
      res += a.e[k][i].real() * b.e[k][i].real() +
             a.e[k][i].imag() * b.e[k][i].imag();
    }
  }
  return res;
}

/**
 * @brief Real part of trace(A * B)
 */
template <typename Real, int Nc>
KOKKOS_INLINE_FUNCTION Real realtraceUV(const MatrixSun<Real, Nc> &a,
                                        const MatrixSun<Real, Nc> &b) {
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
template <typename Real, int Nc>
KOKKOS_INLINE_FUNCTION Real realtraceUVdagger(const MatrixSun<Real, Nc> &a,
                                              const MatrixSun<Real, Nc> &b) {
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
void indexBlock(int block, int &p, int &q) {
  if constexpr (NCOLORS == 3) {
    if (block == 0) {
      p = 0;
      q = 1;
    } else if (block == 1) {
      p = 1;
      q = 2;
    } else {
      p = 0;
      q = 2;
    }
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
template <typename Real, int Nc>
KOKKOS_INLINE_FUNCTION Msu2<Real> getBlockSu2(const MatrixSun<Real, Nc> &tmp1,
                                              int p, int q) {
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
template <typename Real, int Nc>
KOKKOS_INLINE_FUNCTION void mulBlockSun(Msu2<Real> u, MatrixSun<Real, Nc> &link,
                                        int p, int q) {
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

} // namespace kwqft

#endif // KWQFT_MATRIXSUN_HPP
