#pragma once
#include "Space.h"
#include <ratio>

namespace hnswlib {
/**
 * For a given loop unrolling factor K, distance type dist_t, and data type
 * data_t, calculate the L2 squared distance between two vectors. The compiler
 * should automatically do the loop unrolling for us here and vectorize as
 * appropriate.
 */
template <typename dist_t, typename data_t = dist_t, int K = 1,
          typename scalefactor = std::ratio<1, 1>>
static dist_t L2Sqr(const data_t *__restrict pVect1,
                    const data_t *__restrict pVect2, const size_t qty) {
  dist_t res = 0;

  for (size_t i = 0; i < qty / K; i++) {
    for (size_t j = 0; j < K; j++) {
      const size_t index = (i * K) + j;
      dist_t _a = pVect1[index];
      dist_t _b = pVect2[index];
      res += (_a - _b) * (_a - _b);
    }
  }

  constexpr dist_t scale = (dist_t)scalefactor::num / (dist_t)scalefactor::den;
  return (res * scale * scale);
}

template <typename dist_t, typename data_t = dist_t, int K,
          typename scalefactor = std::ratio<1, 1>>
static dist_t L2SqrAtLeast(const data_t *__restrict pVect1,
                           const data_t *__restrict pVect2, const size_t qty) {
  size_t remainder = qty - K;

  return L2Sqr<dist_t, data_t, K, scalefactor>(pVect1, pVect2, K) +
         L2Sqr<dist_t, data_t, 1, scalefactor>(pVect1 + K, pVect2 + K,
                                               remainder);
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float L2SqrSIMD16Ext(const float *pVect1, const float *pVect2,
                            const size_t qty) {
  float PORTABLE_ALIGN64 TmpRes[16];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m512 diff, v1, v2;
  __m512 sum = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    diff = _mm512_sub_ps(v1, v2);
    // sum = _mm512_fmadd_ps(diff, diff, sum);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
  }

  _mm512_store_ps(TmpRes, sum);
  float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] +
              TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] +
              TmpRes[15];

  return (res);
}

#elif defined(USE_AVX)

// Favor using AVX if available.
static float L2SqrSIMD16Ext(const float *pVect1, const float *pVect2,
                            const size_t qty) {
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }

  _mm256_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] +
         TmpRes[6] + TmpRes[7];
}

#elif defined(USE_SSE)

static float L2SqrSIMD16Ext(const float *pVect1, const float *pVect2,
                            const size_t qty) {
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty >> 4;

  const float *pEnd1 = pVect1 + (qty16 << 4);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }

  _mm_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static float L2SqrSIMD16ExtResiduals(const float *pVect1, const float *pVect2,
                                     const size_t qty) {
  size_t qty16 = qty >> 4 << 4;
  float res = L2SqrSIMD16Ext(pVect1, pVect2, qty16);

  size_t qty_left = qty - qty16;
  float res_tail =
      L2Sqr<float, float>(pVect1 + qty16, pVect2 + qty16, qty_left);
  return (res + res_tail);
}
#endif

#ifdef USE_SSE
static float L2SqrSIMD4Ext(const float *pVect1, const float *pVect2,
                           const size_t qty) {
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty4 = qty >> 2;

  const float *pEnd1 = pVect1 + (qty4 << 2);

  __m128 diff, v1, v2;
  __m128 sum = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    diff = _mm_sub_ps(v1, v2);
    sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
  }
  _mm_store_ps(TmpRes, sum);
  return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float L2SqrSIMD4ExtResiduals(const float *pVect1, const float *pVect2,
                                    const size_t qty) {
  size_t qty4 = qty >> 2 << 2;

  float res = L2SqrSIMD4Ext(pVect1, pVect2, qty4);
  size_t qty_left = qty - qty4;

  float res_tail = L2Sqr<float, float>(pVect1 + qty4, pVect2 + qty4, qty_left);

  return (res + res_tail);
}
#endif

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
class EuclideanSpace : public Space<dist_t, data_t> {
  DISTFUNC<dist_t, data_t> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

public:
  EuclideanSpace(size_t dim) : data_size_(dim * sizeof(data_t)), dim_(dim) {
    if (dim % 128 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 128, scalefactor>;
    else if (dim % 64 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 64, scalefactor>;
    else if (dim % 32 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 32, scalefactor>;
    else if (dim % 16 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 16, scalefactor>;
    else if (dim % 8 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 8, scalefactor>;
    else if (dim % 4 == 0)
      fstdistfunc_ = L2Sqr<dist_t, data_t, 4, scalefactor>;

    else if (dim > 128)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 128, scalefactor>;
    else if (dim > 64)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 64, scalefactor>;
    else if (dim > 32)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 32, scalefactor>;
    else if (dim > 16)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 16, scalefactor>;
    else if (dim > 8)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 8, scalefactor>;
    else if (dim > 4)
      fstdistfunc_ = L2SqrAtLeast<dist_t, data_t, 4, scalefactor>;
    else
      fstdistfunc_ = L2Sqr<dist_t, data_t, 1, scalefactor>;
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<dist_t, data_t> get_dist_func() { return fstdistfunc_; }

  size_t get_dist_func_param() { return dim_; }

  ~EuclideanSpace() {}
};

template <>
EuclideanSpace<float, float>::EuclideanSpace(size_t dim)
    : data_size_(dim * sizeof(float)), dim_(dim) {
  fstdistfunc_ = L2Sqr<float, float>;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
  if (dim % 16 == 0)
    fstdistfunc_ = L2SqrSIMD16Ext;
  else if (dim % 4 == 0)
    fstdistfunc_ = L2SqrSIMD4Ext;
  else if (dim > 16)
    fstdistfunc_ = L2SqrSIMD16ExtResiduals;
  else if (dim > 4)
    fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
}
} // namespace hnswlib
