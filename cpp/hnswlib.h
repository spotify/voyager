#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif
#endif

#include "StreamUtils.h"
#include "visited_list_pool.h"
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <string.h>
#include <vector>

namespace hnswlib {
typedef size_t labeltype;

template <typename T> class pairGreater {
public:
  bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
};

template <typename T>
static void writeBinaryPOD(std::shared_ptr<OutputStream> out, const T &podRef) {
  if (!out->write((char *)&podRef, sizeof(T))) {
    throw std::runtime_error("Failed to write " + std::to_string(sizeof(T)) +
                             " bytes to stream!");
  }
}

template <typename T>
static void readBinaryPOD(std::shared_ptr<InputStream> in, T &podRef) {
  long long bytesRead = in->read((char *)&podRef, sizeof(T));
  if (bytesRead != sizeof(T)) {
    throw std::runtime_error("Failed to read " + std::to_string(sizeof(T)) +
                             " bytes from stream! Got " +
                             std::to_string(bytesRead) + ".");
  }
}

template <typename dist_t, typename data_t = dist_t> class AlgorithmInterface {
public:
  virtual void addPoint(const data_t *datapoint, labeltype label) = 0;
  virtual std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const data_t *, size_t, VisitedList *a = nullptr,
            long queryEf = -1) const = 0;

  // Return k nearest neighbor in the order of closer fist
  virtual std::vector<std::pair<dist_t, labeltype>>
  searchKnnCloserFirst(const data_t *query_data, size_t k) const;

  virtual void saveIndex(const std::string &location) = 0;
  virtual ~AlgorithmInterface() {}
};

template <typename dist_t, typename data_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t, data_t>::searchKnnCloserFirst(
    const data_t *query_data, size_t k) const {
  std::vector<std::pair<dist_t, labeltype>> result;

  // here searchKnn returns the result in the order of further first
  auto ret = searchKnn(query_data, k);
  {
    size_t sz = ret.size();
    result.resize(sz);
    while (!ret.empty()) {
      result[--sz] = ret.top();
      ret.pop();
    }
  }

  return result;
}

} // namespace hnswlib

#include "Spaces/Euclidean.h"
#include "Spaces/InnerProduct.h"
#include "hnswalg.h"
