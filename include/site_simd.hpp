/**
 * @file site_simd.hpp
 * @brief Cross-site SIMD gauge-link load/store for OpenMP host builds
 *
 * One Kokkos SIMD lane per lattice site: \c MatrixSun<simd<Real>> holds W
 * independent site matrices. Gauge storage stays interleaved-complex SOA, so
 * element (i, j) of W consecutive links is 2W consecutive reals
 * (re0, im0, re1, im1, ...). When the lanes of a batch address consecutive
 * links, each element is moved with one de-interleaving vector load/store
 * (SVE ld2/st2, NEON ld2/st2, AVX-512 two-source permutes, AVX2
 * unpack/shuffle + cross-lane permute). Row breaks, periodic wraps, halo reads
 * and absent blocks fall back to per-lane access.
 *
 * The ISA branches are optimizations only: any other Kokkos SIMD ABI takes
 * the generic per-lane path, which is correct for every width. Kokkos has no
 * SSE ABI; x86 builds without AVX2 get its scalar ABI (width 1), and the
 * sweeps then use the scalar per-site update.
 */

#ifndef KWQFT_SITE_SIMD_HPP
#define KWQFT_SITE_SIMD_HPP

#include "gauge_load_save.hpp"
#include "matrixsun.hpp"
#include "neighbor_access.hpp"

#ifdef KWQFT_SITE_SIMD

namespace kwqft {

template <typename Real> using SiteSimd = Kokkos::Experimental::simd<Real>;

namespace site_simd_detail {

namespace kx = Kokkos::Experimental;

template <typename Simd, typename Real, typename Abi>
inline constexpr bool is_abi_v =
    std::is_same_v<typename Simd::abi_type, Abi> &&
    std::is_same_v<typename Simd::value_type, Real>;

/// Real and imaginary parts of Simd::size() consecutive complexes at \p p.
template <typename Simd, typename Real>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void
loadContiguous(const Complex<Real> *p, Simd &re, Simd &im) {
  [[maybe_unused]] const Real *r = reinterpret_cast<const Real *>(p);
#if defined(KOKKOS_ARCH_ARM_SVE)
  if constexpr (is_abi_v<Simd, double,
                         kx::simd_abi::sve_fixed_size<SVE_DOUBLES_IN_VECTOR>>) {
    const svfloat64x2_t v = svld2_f64(svptrue_b64(), r);
    re = Simd(static_cast<vls_float64_t>(svget2_f64(v, 0)));
    im = Simd(static_cast<vls_float64_t>(svget2_f64(v, 1)));
    return;
  }
  if constexpr (is_abi_v<Simd, float,
                         kx::simd_abi::sve_fixed_size<SVE_WORDS_IN_VECTOR>>) {
    const svfloat32x2_t v = svld2_f32(svptrue_b32(), r);
    re = Simd(static_cast<vls_float32_t>(svget2_f32(v, 0)));
    im = Simd(static_cast<vls_float32_t>(svget2_f32(v, 1)));
    return;
  }
#endif
#if defined(KOKKOS_ARCH_AVX512XEON)
  if constexpr (is_abi_v<Simd, double, kx::simd_abi::avx512_fixed_size<8>>) {
    const __m512d a = _mm512_loadu_pd(r);
    const __m512d b = _mm512_loadu_pd(r + 8);
    re = Simd(_mm512_permutex2var_pd(
        a, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), b));
    im = Simd(_mm512_permutex2var_pd(
        a, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), b));
    return;
  }
  if constexpr (is_abi_v<Simd, float, kx::simd_abi::avx512_fixed_size<16>>) {
    const __m512 a = _mm512_loadu_ps(r);
    const __m512 b = _mm512_loadu_ps(r + 16);
    re = Simd(_mm512_permutex2var_ps(
        a,
        _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,
                          28, 30),
        b));
    im = Simd(_mm512_permutex2var_ps(
        a,
        _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
                          29, 31),
        b));
    return;
  }
#endif
#if defined(KOKKOS_ARCH_AVX2)
  if constexpr (is_abi_v<Simd, double, kx::simd_abi::avx2_fixed_size<4>>) {
    // unpack within 128-bit halves gives (0 2 1 3); swap the middle pair.
    const __m256d a = _mm256_loadu_pd(r);
    const __m256d b = _mm256_loadu_pd(r + 4);
    re = Simd(_mm256_permute4x64_pd(_mm256_unpacklo_pd(a, b), 0xD8));
    im = Simd(_mm256_permute4x64_pd(_mm256_unpackhi_pd(a, b), 0xD8));
    return;
  }
  if constexpr (is_abi_v<Simd, float, kx::simd_abi::avx2_fixed_size<8>>) {
    // shuffle within 128-bit halves gives pairs (01 45 23 67); swap 45/23.
    const __m256 a = _mm256_loadu_ps(r);
    const __m256 b = _mm256_loadu_ps(r + 8);
    const __m256 e = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256 o = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
    re = Simd(_mm256_castpd_ps(
        _mm256_permute4x64_pd(_mm256_castps_pd(e), 0xD8)));
    im = Simd(_mm256_castpd_ps(
        _mm256_permute4x64_pd(_mm256_castps_pd(o), 0xD8)));
    return;
  }
#endif
#if defined(KOKKOS_ARCH_ARM_NEON)
  if constexpr (is_abi_v<Simd, double, kx::simd_abi::neon_fixed_size<2>>) {
    const float64x2x2_t v = vld2q_f64(r);
    re = Simd(v.val[0]);
    im = Simd(v.val[1]);
    return;
  }
  if constexpr (is_abi_v<Simd, float, kx::simd_abi::neon_fixed_size<4>>) {
    const float32x4x2_t v = vld2q_f32(r);
    re = Simd(v.val[0]);
    im = Simd(v.val[1]);
    return;
  }
#endif
  re = Simd([&](auto l) { return p[static_cast<std::size_t>(l)].x; });
  im = Simd([&](auto l) { return p[static_cast<std::size_t>(l)].y; });
}

/// Inverse of \ref loadContiguous.
template <typename Simd, typename Real>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void
storeContiguous(Complex<Real> *p, const Simd &re, const Simd &im) {
  [[maybe_unused]] Real *r = reinterpret_cast<Real *>(p);
#if defined(KOKKOS_ARCH_ARM_SVE)
  if constexpr (is_abi_v<Simd, double,
                         kx::simd_abi::sve_fixed_size<SVE_DOUBLES_IN_VECTOR>>) {
    svst2_f64(svptrue_b64(), r,
              svcreate2_f64(static_cast<vls_float64_t>(re),
                            static_cast<vls_float64_t>(im)));
    return;
  }
  if constexpr (is_abi_v<Simd, float,
                         kx::simd_abi::sve_fixed_size<SVE_WORDS_IN_VECTOR>>) {
    svst2_f32(svptrue_b32(), r,
              svcreate2_f32(static_cast<vls_float32_t>(re),
                            static_cast<vls_float32_t>(im)));
    return;
  }
#endif
#if defined(KOKKOS_ARCH_AVX512XEON)
  if constexpr (is_abi_v<Simd, double, kx::simd_abi::avx512_fixed_size<8>>) {
    const __m512d a = static_cast<__m512d>(re);
    const __m512d b = static_cast<__m512d>(im);
    _mm512_storeu_pd(r, _mm512_permutex2var_pd(
                            a, _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), b));
    _mm512_storeu_pd(r + 8,
                     _mm512_permutex2var_pd(
                         a, _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15), b));
    return;
  }
  if constexpr (is_abi_v<Simd, float, kx::simd_abi::avx512_fixed_size<16>>) {
    const __m512 a = static_cast<__m512>(re);
    const __m512 b = static_cast<__m512>(im);
    _mm512_storeu_ps(r, _mm512_permutex2var_ps(
                            a,
                            _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20,
                                              5, 21, 6, 22, 7, 23),
                            b));
    _mm512_storeu_ps(r + 16,
                     _mm512_permutex2var_ps(
                         a,
                         _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28,
                                           13, 29, 14, 30, 15, 31),
                         b));
    return;
  }
#endif
#if defined(KOKKOS_ARCH_AVX2)
  if constexpr (is_abi_v<Simd, double, kx::simd_abi::avx2_fixed_size<4>>) {
    const __m256d a =
        _mm256_permute4x64_pd(static_cast<__m256d>(re), 0xD8);
    const __m256d b =
        _mm256_permute4x64_pd(static_cast<__m256d>(im), 0xD8);
    _mm256_storeu_pd(r, _mm256_unpacklo_pd(a, b));
    _mm256_storeu_pd(r + 4, _mm256_unpackhi_pd(a, b));
    return;
  }
  if constexpr (is_abi_v<Simd, float, kx::simd_abi::avx2_fixed_size<8>>) {
    const __m256 a = _mm256_castpd_ps(_mm256_permute4x64_pd(
        _mm256_castps_pd(static_cast<__m256>(re)), 0xD8));
    const __m256 b = _mm256_castpd_ps(_mm256_permute4x64_pd(
        _mm256_castps_pd(static_cast<__m256>(im)), 0xD8));
    _mm256_storeu_ps(r, _mm256_unpacklo_ps(a, b));
    _mm256_storeu_ps(r + 8, _mm256_unpackhi_ps(a, b));
    return;
  }
#endif
#if defined(KOKKOS_ARCH_ARM_NEON)
  if constexpr (is_abi_v<Simd, double, kx::simd_abi::neon_fixed_size<2>>) {
    float64x2x2_t v;
    v.val[0] = static_cast<float64x2_t>(re);
    v.val[1] = static_cast<float64x2_t>(im);
    vst2q_f64(r, v);
    return;
  }
  if constexpr (is_abi_v<Simd, float, kx::simd_abi::neon_fixed_size<4>>) {
    float32x4x2_t v;
    v.val[0] = static_cast<float32x4_t>(re);
    v.val[1] = static_cast<float32x4_t>(im);
    vst2q_f32(r, v);
    return;
  }
#endif
  for (std::size_t l = 0; l < Simd::size(); ++l) {
    p[l] = Complex<Real>(re[l], im[l]);
  }
}

} // namespace site_simd_detail

/**
 * @brief Load the W link matrices addressed by \p ref (one per lane) as one
 *        SIMD matrix, optionally as their Hermitian conjugates.
 *
 * Same element layout and SOA12 handling as \ref loadGaugeLinkRef;
 * \c ref[l].ptr == nullptr gives the zero matrix in lane l.
 */
template <typename Real, typename Simd>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void
loadMatrixBatch(const GaugeLinkRef<Real> *ref, bool adjoint,
                MatrixSun<Simd, NCOLORS> &U) {
  constexpr int W = static_cast<int>(Simd::size());
  const bool soa12 = ref[0].soa12;
  bool contiguous = ref[0].ptr != nullptr;
  for (int l = 1; l < W; ++l) {
    if (ref[l].soa12 != soa12) {
      // Mixed SOA12 / full-SOA lanes: resolve each lane with the scalar path.
      MatrixSun<Real, NCOLORS> m[W];
      for (int k = 0; k < W; ++k) {
        loadGaugeLinkRef(ref[k], adjoint, m[k]);
      }
      for (int i = 0; i < NCOLORS; ++i) {
        for (int j = 0; j < NCOLORS; ++j) {
          U.e[i][j] = Complex<Simd>(
              Simd([&](auto k) {
                return m[static_cast<std::size_t>(k)].e[i][j].x;
              }),
              Simd([&](auto k) {
                return m[static_cast<std::size_t>(k)].e[i][j].y;
              }));
        }
      }
      return;
    }
    contiguous = contiguous && ref[l].ptr == ref[0].ptr + l &&
                 ref[l].stride == ref[0].stride;
  }

  auto fill = [&](auto &&load_elem) {
    Simd re, im;
    if constexpr (NCOLORS == 3) {
      if (soa12) {
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 3; ++j) {
            load_elem(j + i * 3, re, im);
            U.e[i][j] = Complex<Simd>(re, im);
          }
        }
        reconstruct12p(U);
        if (adjoint) {
          U = U.dagger();
        }
        return;
      }
    }
    for (int i = 0; i < NCOLORS; ++i) {
      for (int j = 0; j < NCOLORS; ++j) {
        load_elem(j + i * NCOLORS, re, im);
        if (adjoint) {
          U.e[j][i] = Complex<Simd>(re, -im);
        } else {
          U.e[i][j] = Complex<Simd>(re, im);
        }
      }
    }
  };

  if (contiguous) {
    const Complex<Real> *base = ref[0].ptr;
    const int64_t stride = ref[0].stride;
    fill([&](int e, Simd &re, Simd &im) {
      site_simd_detail::loadContiguous(base + e * stride, re, im);
    });
  } else {
    fill([&](int e, Simd &re, Simd &im) {
      re = Simd([&](auto l) {
        const GaugeLinkRef<Real> &r = ref[static_cast<std::size_t>(l)];
        return r.ptr == nullptr ? Real(0) : r.ptr[e * r.stride].x;
      });
      im = Simd([&](auto l) {
        const GaugeLinkRef<Real> &r = ref[static_cast<std::size_t>(l)];
        return r.ptr == nullptr ? Real(0) : r.ptr[e * r.stride].y;
      });
    });
  }
}

/**
 * @brief Store one SIMD matrix to the links \p link_base[l] (one per lane).
 *
 * Same layout as \ref storeGaugeMatrix (SOA12 writes the first two rows).
 */
template <typename Real, typename Simd>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void
storeMatrixBatch(Complex<Real> *ptr, const int64_t *link_base,
                 int64_t soa_stride, ArrayType atype,
                 const MatrixSun<Simd, NCOLORS> &U) {
  constexpr int W = static_cast<int>(Simd::size());
  bool contiguous = true;
  for (int l = 1; l < W; ++l) {
    contiguous = contiguous && link_base[l] == link_base[0] + l;
  }
  const int rows = (NCOLORS == 3 && atype == ArrayType::SOA12) ? 2 : NCOLORS;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < NCOLORS; ++j) {
      const int64_t off = static_cast<int64_t>(j + i * NCOLORS) * soa_stride;
      if (contiguous) {
        site_simd_detail::storeContiguous(ptr + link_base[0] + off,
                                          U.e[i][j].x, U.e[i][j].y);
      } else {
        for (int l = 0; l < W; ++l) {
          ptr[link_base[l] + off] =
              Complex<Real>(U.e[i][j].x[l], U.e[i][j].y[l]);
        }
      }
    }
  }
}

} // namespace kwqft

#endif // KWQFT_SITE_SIMD

#endif // KWQFT_SITE_SIMD_HPP
