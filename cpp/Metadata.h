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

#include "StreamUtils.h"

namespace voyager {
/**
 * @brief A custom metadata structure used to store index-specific parameters
 * (like number of dimensions in the index, the type of space used, etc).
 *
 * The `Metadata` structure is an abstract parent class, while the nested
 * classes contain logic for setting, accessing, serializing, and deserializing
 * the actual values.
 */
class AbstractMetadata {
public:
  virtual void serializeToStream(std::shared_ptr<OutputStream> outputStream) {
    outputStream->write("VOYA", 4);
    writeBinaryPOD(outputStream, version());
  };
  virtual int version() const = 0;
};

namespace Metadata {
class PreVoyager : public AbstractMetadata {
public:
  PreVoyager(size_t offsetLevel0) : offsetLevel0(offsetLevel0) {}

  virtual int version() const override { return 0; }

  size_t getOffsetLevel0() { return offsetLevel0; }

  virtual void serializeToStream(std::shared_ptr<OutputStream> stream){};

  virtual void loadFromStream(std::shared_ptr<InputStream> stream){};

private:
  size_t offsetLevel0;
};

class V1 : public AbstractMetadata {
public:
  virtual int version() const override { return 1; }

  int getNumDimensions() { return numDimensions; }

  void setNumDimensions(int newNumDimensions) {
    numDimensions = newNumDimensions;
  }

  virtual void serializeToStream(std::shared_ptr<OutputStream> stream) {
    AbstractMetadata::serializeToStream(stream);
    writeBinaryPOD(stream, numDimensions);
  };

  virtual void loadFromStream(std::shared_ptr<InputStream> stream) {
    readBinaryPOD(stream, numDimensions);
  };

private:
  int numDimensions;
};

static std::unique_ptr<AbstractMetadata>
loadFromStream(std::shared_ptr<InputStream> inputStream) {
  // Note: this is 8 on purpose! If the header isn't a match, we
  // read the next four bytes and parse this as a size_t

  char header[8];
  inputStream->read(header, 4);
  bool headerMatch = header[0] == 'V' && header[1] == 'O' && header[2] == 'Y' &&
                     header[3] == 'A';
  if (!headerMatch) {
    inputStream->read(header + 4, 4);
    size_t *offsetLevel0 = (size_t *)&header[0];
    std::unique_ptr<Metadata::PreVoyager> metadata =
        std::make_unique<Metadata::PreVoyager>(*offsetLevel0);
    return metadata;
  }

  int version;
  readBinaryPOD(inputStream, version);

  switch (version) {
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
      error += " A newer version of the Voyager library may be able to "
               "read this "
               "index.";
    } else {
      error += " This index may be corrupted (or not a Voyager index).";
    }

    throw std::domain_error(error);
  }
  }
};

} // namespace Metadata
}; // namespace voyager