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

#include <iostream>
#include <optional>
#include <ratio>
#include <stdlib.h>

#include "Enums.h"
#include "StreamUtils.h"
#include "array_utils.h"
#include "hnswlib.h"

/**
 * A C++ wrapper class for a Voyager index, which accepts
 * and returns floating-point data.
 *
 * This class will be accessed from both Python and Java code,
 * so its interfaces should only include C++ or C datatypes, and
 * it should avoid unnecessary memory copies if possible.
 *
 * The underlying index may use non-floating-point datatypes
 * under-the-hood (i.e.: storing floats as quantized chars)
 * but all methods will only accept and return floats.
 */
class Index {
public:
  virtual ~Index(){};

  virtual void setEF(size_t ef) = 0;
  virtual int getEF() const = 0;

  virtual SpaceType getSpace() const = 0;
  virtual std::string getSpaceName() const = 0;

  virtual StorageDataType getStorageDataType() const = 0;
  virtual std::string getStorageDataTypeName() const = 0;

  virtual int getNumDimensions() const = 0;

  virtual void setNumThreads(int numThreads) = 0;
  virtual int getNumThreads() = 0;

  virtual void saveIndex(const std::string &pathToIndex) = 0;
  virtual void saveIndex(std::shared_ptr<OutputStream> outputStream) = 0;
  virtual void loadIndex(const std::string &pathToIndex,
                         bool searchOnly = false) = 0;
  virtual void loadIndex(std::shared_ptr<InputStream> inputStream,
                         bool searchOnly = false) = 0;

  virtual float getDistance(std::vector<float> a, std::vector<float> b) = 0;

  virtual hnswlib::labeltype addItem(std::vector<float> vector,
                                     std::optional<hnswlib::labeltype> id) = 0;
  virtual std::vector<hnswlib::labeltype>
  addItems(NDArray<float, 2> input, std::vector<hnswlib::labeltype> ids = {},
           int numThreads = -1) = 0;

  virtual std::vector<float> getVector(hnswlib::labeltype id) = 0;
  virtual NDArray<float, 2> getVectors(std::vector<hnswlib::labeltype> ids) = 0;

  virtual std::vector<hnswlib::labeltype> getIDs() const = 0;
  virtual long long getIDsCount() const = 0;
  virtual const std::unordered_map<hnswlib::labeltype, hnswlib::tableint> &
  getIDsMap() const = 0;

  virtual std::tuple<std::vector<hnswlib::labeltype>, std::vector<float>>
  query(std::vector<float> queryVector, int k = 1, long queryEf = -1) = 0;

  virtual std::tuple<NDArray<hnswlib::labeltype, 2>, NDArray<float, 2>>
  query(NDArray<float, 2> queryVectors, int k = 1, int numThreads = -1,
        long queryEf = -1) = 0;

  virtual void markDeleted(hnswlib::labeltype label) = 0;
  virtual void unmarkDeleted(hnswlib::labeltype label) = 0;

  virtual void resizeIndex(size_t newSize) = 0;
  virtual size_t getMaxElements() const = 0;
  virtual size_t getNumElements() const = 0;
  virtual size_t getEfConstruction() const = 0;
  virtual size_t getM() const = 0;
};
