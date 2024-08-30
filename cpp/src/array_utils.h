/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
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

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "E4M3.h"

/**
 * A basic container for an N-dimensional array.
 * Data is stored in a flat std::vector<T>, and the shape of the array is
 * immutable. Can be converted into a Numpy array in Python, or a nested array
 * in Java.
 */
template <typename T, int Dims> class NDArray {
public:
  std::vector<T> data;
  const std::array<int, Dims> shape;
  const std::array<int, Dims> strides;

  NDArray(std::array<int, Dims> shape)
      : data(std::accumulate(shape.begin(), shape.end(), 1,
                             std::multiplies<int>())),
        shape(shape), strides(computeStrides()) {}

  NDArray(std::vector<T> data, std::array<int, Dims> shape)
      : data(data), shape(shape), strides(computeStrides()) {}

  NDArray(T *inputPointer, std::array<int, Dims> shape)
      : data(computeNumElements(shape)), shape(shape),
        strides(computeStrides()) {
    std::memcpy(data.data(), inputPointer, data.size() * sizeof(T));
  }

  T *operator[](int indexInZerothDimension) const {
    return const_cast<T *>(data.data() + (indexInZerothDimension * strides[0]));
  }

private:
  std::array<int, Dims> computeStrides() const {
    std::array<int, Dims> _strides;
    _strides[Dims - 1] = 1;

    for (int i = Dims - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] * shape[i + 1];
    }
    return _strides;
  }

  size_t computeNumElements(std::array<int, Dims> shape) const {
    size_t numOutputElements = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      numOutputElements *= shape[i];
    }
    return numOutputElements;
  }
};

template <typename data_t, typename scalefactor = std::ratio<1, 1>>
NDArray<data_t, 2> floatToDataType(NDArray<float, 2> input) {
  // Handle rescaling to integer storage values if necessary:
  if constexpr (std::is_same_v<data_t, float>) {
    if constexpr (scalefactor::num != scalefactor::den) {
      throw std::runtime_error(
          "Index has a non-unity scale factor set, but is using float32 data "
          "storage. This combination is not yet implemented.");
    }

    return input;
  } else if constexpr (std::is_same_v<data_t, E4M3>) {
    NDArray<E4M3, 2> output(input.shape);

    float *inputPointer = input.data.data();
    E4M3 *outputPointer = output.data.data();

    for (unsigned long i = 0; i < input.data.size(); i++) {
      outputPointer[i] = E4M3(inputPointer[i]);
    }

    return output;
  } else {
    // Re-scale the input values by multiplying by `scalefactor`:
    constexpr float lowerBound = (float)std::numeric_limits<data_t>::min() *
                                 (float)scalefactor::num /
                                 (float)scalefactor::den;
    constexpr float upperBound = (float)std::numeric_limits<data_t>::max() *
                                 (float)scalefactor::num /
                                 (float)scalefactor::den;

    NDArray<data_t, 2> output(input.shape);

    // Re-scale the input values by multiplying by `scalefactor`:
    float *inputPointer = input.data.data();
    data_t *outputPointer = output.data.data();

    for (unsigned long i = 0; i < input.data.size(); i++) {
      if (inputPointer[i] > upperBound || inputPointer[i] < lowerBound) {
        throw std::domain_error(
            "One or more vectors contain values outside of [" +
            std::to_string(lowerBound) + ", " + std::to_string(upperBound) +
            "]. Index: " + std::to_string(i) +
            ", invalid value: " + std::to_string(inputPointer[i]));
      }

      outputPointer[i] =
          (inputPointer[i] * (float)scalefactor::den) / (float)scalefactor::num;
    }

    return output;
  }
}

template <typename data_t, typename scalefactor = std::ratio<1, 1>>
void floatToDataType(const float *inputPointer, data_t *outputPointer,
                     int dimensions) {
  // Handle rescaling to integer storage values if necessary:
  if constexpr (std::is_same_v<data_t, float>) {
    if constexpr (scalefactor::num != scalefactor::den) {
      throw std::runtime_error(
          "Index has a non-unity scale factor set, but is using float32 data "
          "storage. This combination is not yet implemented.");
    }

    std::memcpy(outputPointer, inputPointer, sizeof(float) * dimensions);
  } else if constexpr (std::is_same_v<data_t, E4M3>) {
    // Re-scale the input values by multiplying by `scalefactor`:
    for (int i = 0; i < dimensions; i++) {
      outputPointer[i] = E4M3(inputPointer[i]);
    }
  } else {
    // Re-scale the input values by multiplying by `scalefactor`:
    constexpr float lowerBound = (float)std::numeric_limits<data_t>::min() *
                                 (float)scalefactor::num /
                                 (float)scalefactor::den;
    constexpr float upperBound = (float)std::numeric_limits<data_t>::max() *
                                 (float)scalefactor::num /
                                 (float)scalefactor::den;

    std::vector<data_t> output(dimensions);

    // Re-scale the input values by multiplying by `scalefactor`:
    for (int i = 0; i < dimensions; i++) {
      if (inputPointer[i] > upperBound || inputPointer[i] < lowerBound) {
        throw std::domain_error(
            "One or more vectors contain values outside of [" +
            std::to_string(lowerBound) + ", " + std::to_string(upperBound) +
            "]. Index: " + std::to_string(i) +
            ", invalid value: " + std::to_string(inputPointer[i]));
      }

      outputPointer[i] =
          (inputPointer[i] * (float)scalefactor::den) / (float)scalefactor::num;
    }
  }
}

template <typename data_t, typename scalefactor = std::ratio<1, 1>>
std::vector<data_t> floatToDataType(const std::vector<float> input) {
  if constexpr (std::is_same_v<data_t, float>) {
    if constexpr (scalefactor::num != scalefactor::den) {
      throw std::runtime_error(
          "Index has a non-unity scale factor set, but is using float32 data "
          "storage. This combination is not yet implemented.");
    }

    return input;
  }

  std::vector<data_t> output(input.size());
  floatToDataType<data_t, scalefactor>(input.data(), output.data(),
                                       input.size());
  return output;
}

template <typename data_t, typename scalefactor = std::ratio<1, 1>>
NDArray<float, 2> dataTypeToFloat(NDArray<data_t, 2> input) {
  // Handle rescaling to integer storage values if necessary:
  if constexpr (std::is_same_v<data_t, float>) {
    if constexpr (scalefactor::num != scalefactor::den) {
      throw std::runtime_error(
          "Index has a non-unity scale factor set, but is using float32 data "
          "storage. This combination is not yet implemented.");
    }

    return input;
  } else {
    NDArray<float, 2> output(input.shape);

    // Re-scale the input values by multiplying by `scalefactor`:
    data_t *inputPointer = input.data.data();
    float *outputPointer = output.data.data();

    for (unsigned long i = 0; i < input.data.size(); i++) {
      outputPointer[i] = ((float)inputPointer[i] * (float)scalefactor::num) /
                         (float)scalefactor::den;
    }

    return output;
  }
}

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
void normalizeVector(const float *data, data_t *norm_array, int dimensions) {
  dist_t norm = 0.0;
  for (int i = 0; i < dimensions; i++) {
    if constexpr (scalefactor::num != scalefactor::den) {
      dist_t point = (dist_t)(data[i] * (dist_t)scalefactor::num) /
                     (dist_t)scalefactor::den;
      norm += point * point;
    } else {
      norm += data[i] * data[i];
    }
  }
  norm = 1.0f / (sqrtf(norm) + 1e-30f);
  for (int i = 0; i < dimensions; i++) {
    if constexpr (scalefactor::num != scalefactor::den) {
      dist_t element =
          (data[i] * (dist_t)scalefactor::num) / (dist_t)scalefactor::den;
      dist_t normalizedElement = element * norm;
      norm_array[i] = (normalizedElement * scalefactor::den) / scalefactor::num;
    } else {
      dist_t new_value = data[i] * norm;
      norm_array[i] = new_value;
    }
  }
}

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
dist_t getNorm(const data_t *data, int dimensions) {
  dist_t norm = 0.0;
  for (int i = 0; i < dimensions; i++) {
    if constexpr (scalefactor::num != scalefactor::den) {
      dist_t point = (dist_t)(data[i] * (dist_t)scalefactor::num) /
                     (dist_t)scalefactor::den;
      norm += point * point;
    } else {
      norm += data[i] * data[i];
    }
  }
  return sqrtf(norm);
}

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
bool isNormalized(const data_t *data, int dimensions, dist_t maxNorm) {
  dist_t norm = 0.0;
  for (int i = 0; i < dimensions; i++) {
    if constexpr (scalefactor::num != scalefactor::den) {
      dist_t point = (dist_t)(data[i] * (dist_t)scalefactor::num) /
                     (dist_t)scalefactor::den;
      norm += point * point;
    } else {
      norm += data[i] * data[i];
    }
  }
  return norm <= maxNorm;
}

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
std::string toFloatVectorString(data_t *vec, size_t size) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < size; i++) {
    if constexpr (scalefactor::num != scalefactor::den) {
      float point = (dist_t)(vec[i] * (dist_t)scalefactor::num) /
                    (dist_t)scalefactor::den;
      ss << ((float)point);
    } else {
      ss << ((float)vec[i]);
    }

    if (i < (size - 1)) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
std::string toFloatVectorString(std::vector<data_t> vec) {
  return toFloatVectorString<dist_t, data_t, scalefactor>(vec.data(),
                                                          vec.size());
}

/**
 * Convert a 2D vector of float to NDArray<float, 2>
 */
NDArray<float, 2> vectorsToNDArray(std::vector<std::vector<float>> vectors) {
  int numVectors = vectors.size();
  int dimensions = numVectors > 0 ? vectors[0].size() : 0;
  std::array<int, 2> shape = {numVectors, dimensions};

  // flatten the 2d array into the NDArray's underlying 1D vector
  std::vector<float> flatArray;
  for (const auto &vector : vectors) {
    flatArray.insert(flatArray.end(), vector.begin(), vector.end());
  }

  return NDArray<float, 2>(flatArray, shape);
}
