#pragma once
/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 *
 * This file is heavily based on hnswlib (https://github.com/nmslib/hnswlib,
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

#include "BiMap.h"
#include "Enums.h"
#include "StreamUtils.h"

namespace voyager {
namespace Metadata {
/**
 * @brief A basic metadata class that stores the number of dimensions,
 * the SpaceType, StorageDataType, and number of dimensions.
 */
class V1 {
public:
  V1(int numDimensions, SpaceType spaceType, StorageDataType storageDataType,
     float maxNorm, bool useOrderPreservingTransform)
      : numDimensions(numDimensions), spaceType(spaceType),
        storageDataType(storageDataType), maxNorm(maxNorm),
        useOrderPreservingTransform(useOrderPreservingTransform) {}

  V1() {}
  virtual ~V1() {}

  virtual int version() const { return 1; }

  int getNumDimensions() { return numDimensions; }

  StorageDataType getStorageDataType() { return storageDataType; }

  SpaceType getSpaceType() { return spaceType; }

  float getMaxNorm() { return maxNorm; }

  bool getUseOrderPreservingTransform() const {
    return useOrderPreservingTransform;
  }
  void setUseOrderPreservingTransform(bool newValue) {
    useOrderPreservingTransform = newValue;
  }

  void setNumDimensions(int newNumDimensions) {
    numDimensions = newNumDimensions;
  }

  void setStorageDataType(StorageDataType newStorageDataType) {
    storageDataType = newStorageDataType;
  }

  void setSpaceType(SpaceType newSpaceType) { spaceType = newSpaceType; }

  void setMaxNorm(float newMaxNorm) { maxNorm = newMaxNorm; }

  virtual void serializeToStream(std::shared_ptr<OutputStream> stream) {
    stream->write("VOYA", 4);
    writeBinaryPOD(stream, version());
    writeBinaryPOD(stream, numDimensions);
    writeBinaryPOD(stream, spaceType);
    writeBinaryPOD(stream, storageDataType);
    writeBinaryPOD(stream, maxNorm);
    writeBinaryPOD(stream, useOrderPreservingTransform);
  };

  virtual void loadFromStream(std::shared_ptr<InputStream> stream) {
    // Version has already been loaded before we get here!
    readBinaryPOD(stream, numDimensions);
    readBinaryPOD(stream, spaceType);
    readBinaryPOD(stream, storageDataType);
    readBinaryPOD(stream, maxNorm);
    readBinaryPOD(stream, useOrderPreservingTransform);
  };

private:
  int numDimensions;
  SpaceType spaceType;
  StorageDataType storageDataType;
  float maxNorm;
  bool useOrderPreservingTransform;
};

// File format 2 is emitted by Voyager 3. Voyager 2 only understands format 1
// and rejects this header before attempting to read the graph.
class V2 : public V1 {
public:
  using V1::V1;
  V2() = default;
  int version() const override { return 2; }
  bool stringIndex = false;
  voyager::BiMap names;
  void serializeToStream(std::shared_ptr<OutputStream> stream) override {
    serializeWithNames(stream, stringIndex, names);
  }
  void serializeWithNames(std::shared_ptr<OutputStream> stream,
                          bool isStringIndex,
                          const voyager::BiMap &identifiers) {
    V1::serializeToStream(stream);
    writeBinaryPOD(stream, uint8_t(isStringIndex));
    identifiers.save(stream);
  }
  void loadFromStream(std::shared_ptr<InputStream> stream) override {
    V1::loadFromStream(stream);
    uint8_t kind;
    readBinaryPOD(stream, kind);
    if (kind > 1)
      throw std::domain_error("Invalid Voyager identifier type.");
    stringIndex = kind == 1;
    names = voyager::BiMap::load(stream);
    if (!stringIndex && names.size())
      throw std::domain_error(
          "Numeric Voyager index contains string identifiers.");
  }
};

static std::unique_ptr<Metadata::V1>
loadFromStream(std::shared_ptr<InputStream> inputStream) {
  uint32_t header = inputStream->peek();
  if (header != 'AYOV') {
    return nullptr;
  }

  // Actually read instead of just peeking:
  inputStream->read((char *)&header, sizeof(header));

  int version;
  readBinaryPOD(inputStream, version);

  switch (version) {
  case 2: {
    auto metadata = std::make_unique<Metadata::V2>();
    metadata->loadFromStream(inputStream);
    return metadata;
  }
  case 1: {
    std::unique_ptr<Metadata::V1> metadata = std::make_unique<Metadata::V1>();
    metadata->loadFromStream(inputStream);
    return metadata;
  }
  default: {
    std::stringstream stream;
    stream << std::hex << version;
    std::string resultAsHex(stream.str());

    std::string error = "Unable to parse version of Voyager index file; found "
                        "unsupported version \"0x" +
                        resultAsHex + "\".";

    if (version < 20) {
      error += " A newer version of the Voyager library may be able to read "
               "this index.";
    } else {
      error += " This index may be corrupted (or not a Voyager index).";
    }

    throw std::domain_error(error);
  }
  }
};

} // namespace Metadata
}; // namespace voyager
