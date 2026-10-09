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
#include "TypedIndex.h"

// The string API is shared by all bindings; Index owns and persists the BiMap.
class StringIndex {
public:
  std::shared_ptr<Index> index;

  explicit StringIndex(std::shared_ptr<Index> index) : index(std::move(index)) {
    this->index->enableStringIdentifiers();
  }
  StringIndex(SpaceType space, int dimensions, size_t M = 12,
              size_t efConstruction = 200, size_t randomSeed = 1,
              size_t maxElements = 1,
              StorageDataType storage = StorageDataType::Float32)
      : StringIndex(create(space, dimensions, M, efConstruction, randomSeed,
                           maxElements, storage)) {}

  size_t getNumElements() const { return index->getNumElements(); }
  size_t getMaxElements() const { return index->getMaxElements(); }
  int getNumDimensions() const { return index->getNumDimensions(); }
  std::vector<std::string> getNames() const { return index->getNames(); }
  void resizeIndex(size_t size) { index->resizeIndex(size); }
  void setEF(size_t ef) { index->setEF(ef); }
  void setNumThreads(int threads) { index->setNumThreads(threads); }

  void addItem(const std::string &name, std::vector<float> vector) {
    index->addStringItem(name, std::move(vector));
  }
  void addItems(const std::vector<std::string> &names,
                const std::vector<std::vector<float>> &vectors,
                int numThreads = -1) {
    index->addStringItems(names, vectors, numThreads);
  }
  std::vector<float> getVector(const std::string &name) {
    return index->getVector(index->getStringID(name));
  }
  std::tuple<std::vector<std::string>, std::vector<float>>
  query(std::vector<float> vector, int k = 1, long queryEf = -1) {
    auto result = index->query(std::move(vector), k, queryEf);
    return {index->getNames(std::get<0>(result)), std::get<1>(result)};
  }
  std::tuple<std::vector<std::vector<std::string>>, NDArray<float, 2>>
  query(const std::vector<std::vector<float>> &vectors, int k = 1,
        int numThreads = -1, long queryEf = -1) {
    auto result = index->query(vectors, k, numThreads, queryEf);
    const auto &labels = std::get<0>(result);
    std::vector<std::vector<std::string>> names;
    for (int row = 0; row < labels.shape[0]; ++row) {
      auto begin = labels.data.begin() + row * k;
      names.push_back(
          index->getNames(std::vector<hnswlib::labeltype>(begin, begin + k)));
    }
    return {std::move(names), std::move(std::get<1>(result))};
  }
  void markDeleted(const std::string &name) {
    index->markDeleted(index->getStringID(name));
  }
  void unmarkDeleted(const std::string &name) {
    index->unmarkDeleted(index->getStringID(name));
  }
  void saveIndex(const std::string &path) { index->saveIndex(path); }
  void saveIndex(std::shared_ptr<OutputStream> stream) {
    index->saveIndex(stream);
  }
  static StringIndex load(std::shared_ptr<InputStream> stream) {
    return StringIndex(loadTypedIndexFromStream(stream));
  }
  static StringIndex load(const std::string &path) {
    return load(std::make_shared<FileInputStream>(path));
  }

private:
  static std::shared_ptr<Index> create(SpaceType space, int dimensions,
                                       size_t M, size_t efConstruction,
                                       size_t randomSeed, size_t maxElements,
                                       StorageDataType storage) {
    switch (storage) {
    case StorageDataType::Float32:
      return std::make_shared<TypedIndex<float>>(
          space, dimensions, M, efConstruction, randomSeed, maxElements);
    case StorageDataType::Float8:
      return std::make_shared<TypedIndex<float, int8_t, std::ratio<1, 127>>>(
          space, dimensions, M, efConstruction, randomSeed, maxElements);
    case StorageDataType::E4M3:
      return std::make_shared<TypedIndex<float, E4M3>>(
          space, dimensions, M, efConstruction, randomSeed, maxElements);
    default:
      throw std::domain_error("Unknown storage data type.");
    }
  }
};
