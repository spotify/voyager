#pragma once

#include <atomic>
#include <iostream>
#include <optional>
#include <ratio>

#include "E4M3.h"
#include "Index.h"
#include "array_utils.h"
#include "hnswlib.h"
#include "std_utils.h"

template <typename T> inline const StorageDataType storageDataType();
template <typename T> inline const std::string storageDataTypeName();

template <>
const StorageDataType storageDataType<int8_t>() {
  return StorageDataType::Float8;
}
template <> const StorageDataType storageDataType<float>() {
  return StorageDataType::Float32;
}
template <> const StorageDataType storageDataType<struct E4M3>() {
  return StorageDataType::E4M3;
}

template <>
const std::string storageDataTypeName<int8_t>() {
  return "Float8";
}
template <>
const std::string storageDataTypeName<float>() {
  return "Float32";
}
template <>
const std::string storageDataTypeName<struct E4M3>() {
  return "E4M3";
}

template <typename dist_t, typename data_t>
dist_t ensureNotNegative(dist_t distance, hnswlib::labeltype label) {
  if constexpr (std::is_same_v<data_t, struct E4M3>) {
    // Allow for a very slight negative distance if using E4M3
    if (distance < 0 && distance >= -0.14) {
      return 0;
    }
  }

  if (distance < 0) {
    if (distance >= -0.00001) {
      return 0;
    }

    throw std::runtime_error(
        "Potential candidate (with label '" + std::to_string(label) +
        "') had negative distance " + std::to_string(distance) +
        ". This may indicate a corrupted index file.");
  }

  return distance;
}

/**
 * A C++ wrapper class for a typed HNSW index.
 *
 * This class will be accessed from both Python and Java code,
 * so its interfaces should only include C++ or C datatypes, and
 * it should avoid unnecessary memory copies if possible.
 */
template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
class TypedIndex : public Index {
private:
  static const int ser_version = 1; // serialization version

  SpaceType space;
  int dimensions;

  size_t seed;
  size_t defaultEF;

  bool ep_added;
  bool normalize = false;
  int numThreadsDefault;
  hnswlib::labeltype currentLabel;
  std::unique_ptr<hnswlib::HierarchicalNSW<dist_t, data_t>> algorithmImpl;
  std::unique_ptr<hnswlib::Space<dist_t, data_t>> spaceImpl;

public:
  /**
   * Create an empty index with the given parameters.
   */
  TypedIndex(const SpaceType space, const int dimensions, const size_t M = 12,
             const size_t efConstruction = 200, const size_t randomSeed = 1,
             const size_t maxElements = 1)
      : space(space), dimensions(dimensions) {
    switch (space) {
    case Euclidean:
      spaceImpl = std::make_unique<
          hnswlib::EuclideanSpace<dist_t, data_t, scalefactor>>(dimensions);
      break;
    case InnerProduct:
      spaceImpl = std::make_unique<
          hnswlib::InnerProductSpace<dist_t, data_t, scalefactor>>(dimensions);
      break;
    case Cosine:
      spaceImpl = std::make_unique<
          hnswlib::InnerProductSpace<dist_t, data_t, scalefactor>>(dimensions);
      normalize = true;
      break;
    default:
      throw new std::runtime_error(
          "Space must be one of Euclidean, InnerProduct, or Cosine.");
    }

    ep_added = true;
    numThreadsDefault = std::thread::hardware_concurrency();

    defaultEF = 10;

    currentLabel = 0;
    algorithmImpl = std::make_unique<hnswlib::HierarchicalNSW<dist_t, data_t>>(
        spaceImpl.get(), maxElements, M, efConstruction, randomSeed);

    ep_added = false;
    algorithmImpl->ef_ = defaultEF;

    seed = randomSeed;
  }

  virtual ~TypedIndex() {}

  /**
   * Load an index from the given .hnsw file on disk, interpreting
   * it as the given Space and number of dimensions.
   */
  TypedIndex(const std::string &indexFilename, const SpaceType space,
             const int dimensions, bool searchOnly = false)
      : TypedIndex(space, dimensions) {
    algorithmImpl = std::make_unique<hnswlib::HierarchicalNSW<dist_t, data_t>>(
        spaceImpl.get(), indexFilename, 0, searchOnly);
    currentLabel = algorithmImpl->cur_element_count;
  }

  /**
   * Load an index from the given input stream, interpreting
   * it as the given Space and number of dimensions.
   */
  TypedIndex(std::shared_ptr<InputStream> inputStream, const SpaceType space,
             const int dimensions, bool searchOnly = false)
      : TypedIndex(space, dimensions) {
    algorithmImpl = std::make_unique<hnswlib::HierarchicalNSW<dist_t, data_t>>(
        spaceImpl.get(), inputStream, 0, searchOnly);
    currentLabel = algorithmImpl->cur_element_count;
  }

  int getNumDimensions() const { return dimensions; }

  SpaceType getSpace() const { return space; }

  std::string getSpaceName() const {
    // TODO: Use magic_enum?
    switch (space) {
    case SpaceType::Euclidean:
      return "Euclidean";
    case SpaceType::InnerProduct:
      return "InnerProduct";
    case SpaceType::Cosine:
      return "Cosine";
    default:
      return "unknown";
    }
  }

  StorageDataType getStorageDataType() const {
    return storageDataType<data_t>();
  }

  std::string getStorageDataTypeName() const {
    return storageDataTypeName<data_t>();
  }

  void setEF(size_t ef) {
    defaultEF = ef;
    if (algorithmImpl)
      algorithmImpl->ef_ = ef;
  }

  void setNumThreads(int numThreads) { numThreadsDefault = numThreads; }

  void loadIndex(const std::string &pathToIndex, bool searchOnly = false) {
    throw std::runtime_error("Not implemented.");
  }

  void loadIndex(std::shared_ptr<InputStream> inputStream,
                 bool searchOnly = false) {
    throw std::runtime_error("Not implemented.");
  }

  /**
   * Save this index to the provided file path on disk.
   */
  void saveIndex(const std::string &pathToIndex) {
    algorithmImpl->saveIndex(pathToIndex);
  }

  /**
   * Save this HNSW index file to the provided output stream.
   * The bytes written to the given output stream can be passed to the
   * TypedIndex constructor to reload this index.
   */
  void saveIndex(std::shared_ptr<OutputStream> outputStream) {
    algorithmImpl->saveIndex(outputStream);
  }

  float getDistance(std::vector<float> _a, std::vector<float> _b) {
    std::vector<data_t> a(dimensions);
    std::vector<data_t> b(dimensions);

    if (_a.size() != dimensions || _b.size() != dimensions) {
      throw std::runtime_error("Index has " + std::to_string(dimensions) +
                               " dimensions, but received vectors of size: " +
                               std::to_string(_a.size()) + " and " +
                               std::to_string(_b.size()) + ".");
    }

    if (normalize) {
      normalizeVector<dist_t, data_t, scalefactor>(_a.data(), a.data(),
                                                   a.size());
      normalizeVector<dist_t, data_t, scalefactor>(_b.data(), b.data(),
                                                   b.size());
    } else {
      floatToDataType<data_t, scalefactor>(_a.data(), a.data(), a.size());
      floatToDataType<data_t, scalefactor>(_b.data(), b.data(), b.size());
    }

    return spaceImpl->get_dist_func()(a.data(), b.data(), dimensions);
  }

  void addItem(std::vector<float> vector, std::optional<size_t> id) {
    std::vector<size_t> ids;

    if (id) {
      ids.push_back(*id);
    }

    addItems(NDArray<float, 2>(vector, {1, (int)vector.size()}), ids);
  }

  void addItems(NDArray<float, 2> floatInput, std::vector<size_t> ids = {},
                int numThreads = -1) {
    if (numThreads <= 0)
      numThreads = numThreadsDefault;

    size_t rows = std::get<0>(floatInput.shape);
    size_t features = std::get<1>(floatInput.shape);

    if (features != (size_t)dimensions)
      throw std::runtime_error("Provided features per vector does not match "
                               "this index's dimensionality.");

    // avoid using threads when the number of searches is small:
    if (rows <= ((size_t)numThreads * 4)) {
      numThreads = 1;
    }

    if (!ids.empty() && (unsigned long)ids.size() != rows) {
      throw std::runtime_error("Provided IDs don't match number of vectors.");
    }

    // TODO: Should we always double the number of elements instead? Maybe use
    // an adaptive algorithm to minimize both reallocations and memory usage?
    if (getNumElements() + rows > getMaxElements()) {
      resizeIndex(getNumElements() + rows);
    }

    int start = 0;
    if (!ep_added) {
      size_t id = ids.size() ? ids.at(0) : (currentLabel);
      std::vector<data_t> convertedVector(dimensions);

      if (normalize) {
        normalizeVector<dist_t, data_t, scalefactor>(
            floatInput[0], convertedVector.data(), convertedVector.size());
      } else {
        floatToDataType<data_t, scalefactor>(
            floatInput[0], convertedVector.data(), convertedVector.size());
      }

      algorithmImpl->addPoint(convertedVector.data(), (size_t)id);
      start = 1;
      ep_added = true;
    }

    if (!normalize) {
      std::vector<data_t> convertedArray(numThreads * dimensions);
      ParallelFor(start, rows, numThreads, [&](size_t row, size_t threadId) {
        size_t startIndex = threadId * dimensions;
        floatToDataType<data_t, scalefactor>(
            floatInput[row], (convertedArray.data() + startIndex), dimensions);
        size_t id = ids.size() ? ids.at(row) : (currentLabel + row);
        algorithmImpl->addPoint(convertedArray.data() + startIndex, id);
      });
    } else {
      std::vector<data_t> normalizedArray(numThreads * dimensions);
      ParallelFor(start, rows, numThreads, [&](size_t row, size_t threadId) {
        size_t startIndex = threadId * dimensions;
        normalizeVector<dist_t, data_t, scalefactor>(
            floatInput[row], (normalizedArray.data() + startIndex), dimensions);
        size_t id = ids.size() ? ids.at(row) : (currentLabel + row);
        algorithmImpl->addPoint(normalizedArray.data() + startIndex, id);
      });
    };

    currentLabel += rows;
  }

  std::vector<data_t> getRawVector(hnswlib::labeltype id) {
    return algorithmImpl->getDataByLabel(id);
  }

  std::vector<float> getVector(hnswlib::labeltype id) {
    std::vector<data_t> rawData = getRawVector(id);
    NDArray<data_t, 2> output(rawData, {1, (int)rawData.size()});
    return dataTypeToFloat<data_t, scalefactor>(output).data;
  }

  NDArray<float, 2> getVectors(std::vector<hnswlib::labeltype> ids) {
    NDArray<float, 2> output = NDArray<float, 2>({(int)ids.size(), dimensions});

    for (unsigned long i = 0; i < ids.size(); i++) {
      std::vector<float> vector = getVector(ids[i]);
      std::copy(vector.begin(), vector.end(),
                output.data.data() + (i * dimensions));
    }

    return output;
  }

  std::vector<hnswlib::labeltype> getIDs() const {
    std::vector<hnswlib::labeltype> ids;
    ids.reserve(algorithmImpl->label_lookup_.size());

    for (auto const &kv : algorithmImpl->label_lookup_) {
      ids.push_back(kv.first);
    }

    return ids;
  }

  long long getIDsCount() const { return algorithmImpl->label_lookup_.size(); }

  const std::unordered_map<hnswlib::labeltype, hnswlib::tableint> &
  getIDsMap() const {
    return algorithmImpl->label_lookup_;
  }

  std::tuple<NDArray<hnswlib::labeltype, 2>, NDArray<dist_t, 2>>
  query(NDArray<float, 2> floatQueryVectors, int k = 1, int numThreads = -1,
        long queryEf = -1) {
    if (queryEf > 0 && queryEf < k) {
      throw std::runtime_error("queryEf must be equal to or greater than the requested number of neighbors");
    }
    int numRows = std::get<0>(floatQueryVectors.shape);
    int numFeatures = std::get<1>(floatQueryVectors.shape);

    if (numFeatures != dimensions) {
      throw std::runtime_error(
          "Query vectors expected to share dimensionality with index.");
    }

    NDArray<hnswlib::labeltype, 2> labels({numRows, k});
    NDArray<dist_t, 2> distances({numRows, k});

    hnswlib::labeltype *labelPointer = labels.data.data();
    dist_t *distancePointer = distances.data.data();

    if (numThreads <= 0) {
      numThreads = numThreadsDefault;
    }

    // avoid using threads when the number of searches is small:

    if (numRows <= numThreads * 4) {
      numThreads = 1;
    }

    if (normalize == false) {
      std::vector<data_t> convertedArray(numThreads * numFeatures);
      ParallelFor(0, numRows, numThreads, [&](size_t row, size_t threadId) {
        size_t start_idx = threadId * dimensions;
        floatToDataType<data_t, scalefactor>(
            floatQueryVectors[row], (convertedArray.data() + start_idx),
            dimensions);

        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result =
            algorithmImpl->searchKnn((convertedArray.data() + start_idx), k,
                                     nullptr, queryEf);

        if (result.size() != (unsigned long)k) {
          throw std::runtime_error(
              "Fewer than expected results were retrieved; only found " +
              std::to_string(result.size()) + " of " + std::to_string(k) +
              " requested neighbors.");
        }

        for (int i = k - 1; i >= 0; i--) {
          auto &result_tuple = result.top();

          dist_t distance = result_tuple.first;

          distancePointer[row * k + i] = distance;
          labelPointer[row * k + i] = result_tuple.second;
          result.pop();
        }
      });
    } else {
      std::vector<data_t> norm_array(numThreads * numFeatures);
      ParallelFor(0, numRows, numThreads, [&](size_t row, size_t threadId) {
        size_t start_idx = threadId * dimensions;
        normalizeVector<dist_t, data_t, scalefactor>(
            floatQueryVectors[row], (norm_array.data() + start_idx),
            dimensions);

        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result =
            algorithmImpl->searchKnn(norm_array.data() + start_idx, k, nullptr,
                                     queryEf);

        if (result.size() != (unsigned long)k) {
          throw std::runtime_error(
              "Fewer than expected results were retrieved; only found " +
              std::to_string(result.size()) + " of " + std::to_string(k) +
              " requested neighbors.");
        }

        for (int i = k - 1; i >= 0; i--) {
          auto &result_tuple = result.top();

          dist_t distance = result_tuple.first;
          distancePointer[row * k + i] = ensureNotNegative<dist_t, data_t>(
              result_tuple.first, result_tuple.second);
          labelPointer[row * k + i] = result_tuple.second;
          result.pop();
        }
      });
    }

    return {labels, distances};
  }

  std::tuple<std::vector<hnswlib::labeltype>, std::vector<float>>
  query(std::vector<float> floatQueryVector, int k = 1, long queryEf = -1) {
    if (queryEf > 0 && queryEf < k) {
      throw std::runtime_error("queryEf must be equal to or greater than the requested number of neighbors");
    }

    int numFeatures = floatQueryVector.size();

    if (numFeatures != dimensions) {
      throw std::runtime_error(
          "Query vector expected to share dimensionality with index.");
    }

    std::vector<hnswlib::labeltype> labels(k);
    std::vector<dist_t> distances(k);

    hnswlib::labeltype *labelPointer = labels.data();
    dist_t *distancePointer = distances.data();

    if (normalize == false) {
      const std::vector<data_t> queryVector =
          floatToDataType<data_t, scalefactor>(floatQueryVector);

      std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result =
          algorithmImpl->searchKnn(queryVector.data(), k, nullptr, queryEf);

      if (result.size() != (unsigned long)k) {
        throw std::runtime_error(
            "Fewer than expected results were retrieved; only found " +
            std::to_string(result.size()) + " of " + std::to_string(k) +
            " requested neighbors.");
      }

      for (int i = k - 1; i >= 0; i--) {
        auto &result_tuple = result.top();
        distancePointer[i] = result_tuple.first;
        labelPointer[i] = result_tuple.second;
        result.pop();
      }
    } else {
      std::vector<data_t> norm_array(numFeatures);
      normalizeVector<dist_t, data_t, scalefactor>(
          floatQueryVector.data(), norm_array.data(), dimensions);

      std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result =
          algorithmImpl->searchKnn(norm_array.data(), k, nullptr, queryEf);

      if (result.size() != (unsigned long)k) {
        throw std::runtime_error(
            "Fewer than expected results were retrieved; only found " +
            std::to_string(result.size()) + " of " + std::to_string(k) +
            " requested neighbors.");
      }

      for (int i = k - 1; i >= 0; i--) {
        auto &result_tuple = result.top();

        distancePointer[i] = ensureNotNegative<dist_t, data_t>(
            result_tuple.first, result_tuple.second);
        labelPointer[i] = result_tuple.second;
        result.pop();
      }
    }

    return {labels, distances};
  }

  void markDeleted(hnswlib::labeltype label) {
    algorithmImpl->markDelete(label);
  }

  void unmarkDeleted(hnswlib::labeltype label) {
    algorithmImpl->unmarkDelete(label);
  }

  void resizeIndex(size_t new_size) { algorithmImpl->resizeIndex(new_size); }

  size_t getMaxElements() const { return algorithmImpl->max_elements_; }

  size_t getNumElements() const { return algorithmImpl->cur_element_count; }

  int getEF() const { return defaultEF; }

  int getNumThreads() { return numThreadsDefault; }

  size_t getEfConstruction() const { return algorithmImpl->ef_construction_; }

  size_t getM() const { return algorithmImpl->M_; }
};
