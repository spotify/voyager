#pragma once
#include "InnerProduct.h"
#include "Space.h"
#include <cmath>
#include <ratio>

namespace hnswlib {

template <typename dist_t, typename data_t = dist_t, int K = 1,
          typename scalefactor = std::ratio<1, 1>>
static dist_t Cosine(const data_t *pVect1, const data_t *pVect2, size_t qty) {
  dist_t res = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
      pVect1, pVect2, qty);

  dist_t magSquared1 = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
      pVect1, pVect1, qty);
  dist_t magSquared2 = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
      pVect2, pVect2, qty);
  dist_t denominator = sqrtf(magSquared1) * sqrtf(magSquared2);

  constexpr dist_t scale = (dist_t)scalefactor::num / (dist_t)scalefactor::den;
  res *= scale * scale;

  res /= denominator;

  res = (static_cast<dist_t>(1.0f) - res);
  return res;
}

template <typename dist_t, typename data_t = dist_t, int K,
          typename scalefactor = std::ratio<1, 1>>
static dist_t CosineAtLeast(const data_t *__restrict pVect1,
                            const data_t *__restrict pVect2, const size_t qty) {
  size_t remainder = qty - K;
  dist_t res = InnerProductWithoutScale<dist_t, data_t, K, scalefactor>(
                   pVect1, pVect2, K) +
               InnerProductWithoutScale<dist_t, data_t, 1, scalefactor>(
                   pVect1 + K, pVect2 + K, remainder);

  dist_t magSquared1 =
      InnerProductAtLeast<dist_t, data_t, K, scalefactor>(pVect1, pVect1, qty);
  dist_t magSquared2 =
      InnerProductAtLeast<dist_t, data_t, K, scalefactor>(pVect2, pVect2, qty);
  dist_t denominator = sqrtf(magSquared1) * sqrtf(magSquared2);

  constexpr dist_t scale = (dist_t)scalefactor::num / (dist_t)scalefactor::den;
  res *= scale * scale;

  res /= denominator;

  res = (static_cast<dist_t>(1.0f) - res);
  return res;
}

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
class CosineSpace : public Space<dist_t, data_t> {
  DISTFUNC<dist_t, data_t> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

public:
  CosineSpace(size_t dim) : data_size_(dim * sizeof(data_t)), dim_(dim) {
    if (dim % 128 == 0)
      fstdistfunc_ = Cosine<dist_t, data_t, 128, scalefactor>;
    else if (dim % 64 == 0)
      fstdistfunc_ = Cosine<dist_t, data_t, 64, scalefactor>;
    else if (dim % 32 == 0)
      fstdistfunc_ = Cosine<dist_t, data_t, 32, scalefactor>;
    else if (dim % 16 == 0)
      fstdistfunc_ = Cosine<dist_t, data_t, 16, scalefactor>;
    else if (dim % 8 == 0)
      fstdistfunc_ = Cosine<dist_t, data_t, 8, scalefactor>;
    else if (dim % 4 == 0)
      fstdistfunc_ = Cosine<dist_t, data_t, 4, scalefactor>;

    else if (dim > 128)
      fstdistfunc_ = CosineAtLeast<dist_t, data_t, 128, scalefactor>;
    else if (dim > 64)
      fstdistfunc_ = CosineAtLeast<dist_t, data_t, 64, scalefactor>;
    else if (dim > 32)
      fstdistfunc_ = CosineAtLeast<dist_t, data_t, 32, scalefactor>;
    else if (dim > 16)
      fstdistfunc_ = CosineAtLeast<dist_t, data_t, 16, scalefactor>;
    else if (dim > 8)
      fstdistfunc_ = CosineAtLeast<dist_t, data_t, 8, scalefactor>;
    else if (dim > 4)
      fstdistfunc_ = CosineAtLeast<dist_t, data_t, 4, scalefactor>;
    else
      fstdistfunc_ = Cosine<dist_t, data_t, 1, scalefactor>;
  }

  size_t get_data_size() { return data_size_; }

  DISTFUNC<dist_t, data_t> get_dist_func() { return fstdistfunc_; }

  size_t get_dist_func_param() { return dim_; }
  ~CosineSpace() {}
};

} // namespace hnswlib