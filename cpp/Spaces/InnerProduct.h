/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 *
 * This file is includes code from hnswlib (https://github.com/nmslib/hnswlib,
 * Apache 2.0-licensed, no copyright author listed)
 * --
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * -/-/-
 */

#pragma once
#include "Space.h"
#include <ratio>

namespace hnswlib {
/**
 * For a given loop unrolling factor K, distance type dist_t, and data type
 * data_t, calculate the inner product distance between two vectors. The
 * compiler should automatically do the loop unrolling for us here and vectorize
 * as appropriate.
 */
template <typename dist_t, typename data_t = dist_t, int K = 1,
          typename scalefactor = std::ratio<1, 1>>
static dist_t InnerProductWithoutScale(const data_t *pVect1,
                                       const data_t *pVect2, size_t qty) {
  dist_t res = 0;

  qty = qty / K;

  for (size_t i = 0; i < qty; i++) {
    for (size_t j = 0; j < K; j++) {
      const size_t index = (i * K) + j;
      dist_t _a = pVect1[index];
      dist_t _b = pVect2[index];
      res += _a * _b;
    }
  }
  return res;
}

template <typename dist_t, typename data_t = dist_t, int K = 1,
          typename scalefactor = std::ratio<1, 1>>
static dist_t InnerProduct(const data_t *pVect1, const data_t *pVect2,
                           size_t qty) {
  dist_t res = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
      pVect1, pVect2, qty);
  constexpr dist_t scale = (dist_t)scalefactor::num / (dist_t)scalefactor::den;
  res *= scale * scale;
  res = (static_cast<dist_t>(1.0f) - res);
  return res;
}

template <typename dist_t, typename data_t = dist_t, int K,
          typename scalefactor = std::ratio<1, 1>>
static dist_t InnerProductAtLeast(const data_t *__restrict pVect1,
                                  const data_t *__restrict pVect2,
                                  const size_t qty) {
  size_t remainder = qty - K;
  dist_t res = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
                   pVect1, pVect2, K) +
               InnerProductWithoutScale<dist_t, data_t, 1, scalefactor>(
                   pVect1 + K, pVect2 + K, remainder);
  constexpr dist_t scale = (dist_t)scalefactor::num / (dist_t)scalefactor::den;
  res *= scale * scale;
  return res;
}

template <typename dist_t, typename data_t = dist_t, int K,
          typename scalefactor = std::ratio<1, 1>>
static dist_t InnerProductDistanceAtLeast(const data_t *__restrict pVect1,
                                          const data_t *__restrict pVect2,
                                          const size_t qty) {
  return (
      static_cast<dist_t>(1.0f) -
      InnerProductAtLeast<dist_t, data_t, K, scalefactor>(pVect1, pVect2, qty));
}

#if defined(USE_AVX)

// Favor using AVX if available.
static float InnerProductSIMD4Ext(const float *pVect1, const float *pVect2,
                                  const size_t qty) {
  float PORTABLE_ALIGN32 TmpRes[8];

  size_t qty16 = qty / 16;
  size_t qty4 = qty / 4;

  const float *pEnd1 = pVect1 + 16 * qty16;
  const float *pEnd2 = pVect1 + 4 * qty4;

  __m256 sum256 = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
  }

  __m128 v1, v2;
  __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0),
                               _mm256_extractf128_ps(sum256, 1));

  while (pVect1 < pEnd2) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
  }

  _mm_store_ps(TmpRes, sum_prod);
  float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
  ;
  return 1.0f - sum;
}

#elif defined(USE_SSE)

static float InnerProductSIMD4Ext(const float *pVect1, const float *pVect2,
                                  const size_t qty) {
  float PORTABLE_ALIGN32 TmpRes[8];

  size_t qty16 = qty / 16;
  size_t qty4 = qty / 4;

  const float *pEnd1 = pVect1 + 16 * qty16;
  const float *pEnd2 = pVect1 + 4 * qty4;

  __m128 v1, v2;
  __m128 sum_prod = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
  }

  while (pVect1 < pEnd2) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
  }

  _mm_store_ps(TmpRes, sum_prod);
  float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

  return 1.0f - sum;
}

#endif

#if defined(USE_AVX512)

static float InnerProductSIMD16Ext(const float *pVect1, const float *pVect2,
                                   const size_t qty) {
  float PORTABLE_ALIGN64 TmpRes[16];

  size_t qty16 = qty / 16;

  const float *pEnd1 = pVect1 + 16 * qty16;

  __m512 sum512 = _mm512_set1_ps(0);

  while (pVect1 < pEnd1) {
    //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

    __m512 v1 = _mm512_loadu_ps(pVect1);
    pVect1 += 16;
    __m512 v2 = _mm512_loadu_ps(pVect2);
    pVect2 += 16;
    sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
  }

  _mm512_store_ps(TmpRes, sum512);
  float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] +
              TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] +
              TmpRes[15];

  return 1.0f - sum;
}

#elif defined(USE_AVX)

static float InnerProductSIMD16Ext(const float *pVect1, const float *pVect2,
                                   const size_t qty) {
  float PORTABLE_ALIGN32 TmpRes[8];

  size_t qty16 = qty / 16;

  const float *pEnd1 = pVect1 + 16 * qty16;

  __m256 sum256 = _mm256_set1_ps(0);

  while (pVect1 < pEnd1) {
    //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

    __m256 v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    __m256 v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

    v1 = _mm256_loadu_ps(pVect1);
    pVect1 += 8;
    v2 = _mm256_loadu_ps(pVect2);
    pVect2 += 8;
    sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
  }

  _mm256_store_ps(TmpRes, sum256);
  float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7];

  return 1.0f - sum;
}

#elif defined(USE_SSE)

static float InnerProductSIMD16Ext(const float *pVect1, const float *pVect2,
                                   const size_t qty) {
  float PORTABLE_ALIGN32 TmpRes[8];
  size_t qty16 = qty / 16;

  const float *pEnd1 = pVect1 + 16 * qty16;

  __m128 v1, v2;
  __m128 sum_prod = _mm_set1_ps(0);

  while (pVect1 < pEnd1) {
    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

    v1 = _mm_loadu_ps(pVect1);
    pVect1 += 4;
    v2 = _mm_loadu_ps(pVect2);
    pVect2 += 4;
    sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
  }
  _mm_store_ps(TmpRes, sum_prod);
  float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

  return 1.0f - sum;
}

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static float InnerProductSIMD16ExtResiduals(const float *pVect1,
                                            const float *pVect2,
                                            const size_t qty) {
  size_t qty16 = qty >> 4 << 4;
  float res = InnerProductSIMD16Ext(pVect1, pVect2, qty16);

  size_t qty_left = qty - qty16;
  float res_tail =
      InnerProduct<float, float>(pVect1 + qty16, pVect2 + qty16, qty_left);
  return res + res_tail - 1.0f;
}

static float InnerProductSIMD4ExtResiduals(const float *pVect1,
                                           const float *pVect2,
                                           const size_t qty) {
  size_t qty4 = qty >> 2 << 2;

  float res = InnerProductSIMD4Ext(pVect1, pVect2, qty4);
  size_t qty_left = qty - qty4;

  float res_tail =
      InnerProduct<float, float>(pVect1 + qty4, pVect2 + qty4, qty_left);

  return res + res_tail - 1.0f;
}
#endif

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
class InnerProductSpace : public Space<dist_t, data_t> {
  DISTFUNC<dist_t, data_t> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

public:
  InnerProductSpace(size_t dim) : data_size_(dim * sizeof(data_t)), dim_(dim) {
    if (dim % 128 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 128, scalefactor>;
    else if (dim % 64 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 64, scalefactor>;
    else if (dim % 32 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 32, scalefactor>;
    else if (dim % 16 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 16, scalefactor>;
    else if (dim % 8 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 8, scalefactor>;
    else if (dim % 4 == 0)
      fstdistfunc_ = InnerProduct<dist_t, data_t, 4, scalefactor>;

    else if (dim > 128)
      fstdistfunc_ =
          InnerProductDistanceAtLeast<dist_t, data_t, 128, scalefactor>;
    else if (dim > 64)
      fstdistfunc_ =
          InnerProductDistanceAtLeast<dist_t, data_t, 64, scalefactor>;
    else if (dim > 32)
      fstdistfunc_ =
          InnerProductDistanceAtLeast<dist_t, data_t, 32, scalefactor>;
    else if (dim > 16)
      fstdistfunc_ =
          InnerProductDistanceAtLeast<dist_t, data_t, 16, scalefactor>;
    else if (dim > 8)
      fstdistfunc_ =
          InnerProductDistanceAtLeast<dist_t, data_t, 8, scalefactor>;
    else if (dim > 4)
      fstdistfunc_ =
          InnerProductDistanceAtLeast<dist_t, data_t, 4, scalefactor>;
    else
      fstdistfunc_ = InnerProduct<dist_t, data_t, 1, scalefactor>;
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<dist_t, data_t> get_dist_func() { return fstdistfunc_; }

  size_t get_dist_func_param() { return dim_; }
  ~InnerProductSpace() {}
};

template <>
InnerProductSpace<float, float>::InnerProductSpace(size_t dim)
    : data_size_(dim * sizeof(float)), dim_(dim) {
  fstdistfunc_ = InnerProduct<float, float>;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
  if (dim % 16 == 0)
    fstdistfunc_ = InnerProductSIMD16Ext;
  else if (dim % 4 == 0)
    fstdistfunc_ = InnerProductSIMD4Ext;
  else if (dim > 16)
    fstdistfunc_ = InnerProductSIMD16ExtResiduals;
  else if (dim > 4)
    fstdistfunc_ = InnerProductSIMD4ExtResiduals;
#endif
}

} // namespace hnswlib
