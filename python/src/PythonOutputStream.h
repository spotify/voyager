// Copyright 2022-2023 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>
#include <optional>

namespace nb = nanobind;

#include "../../cpp/src/StreamUtils.h"
#include "PythonFileLike.h"

bool isWriteableFileLike(nb::object fileLike) {
  return nb::hasattr(fileLike, "write") && nb::hasattr(fileLike, "seek") &&
         nb::hasattr(fileLike, "tell") && nb::hasattr(fileLike, "seekable");
}

/**
 * An OutputStream subclass that writes its
 * data to a provided Python file-like object.
 */
class PythonOutputStream : public OutputStream, public PythonFileLike {
public:
  static constexpr unsigned long long MAX_BUFFER_SIZE = 1024 * 1024 * 100;

  PythonOutputStream(nb::object fileLike) : PythonFileLike(fileLike) {
    if (!isWriteableFileLike(fileLike)) {
      throw nb::type_error("Expected a file-like object (with write, seek, "
                           "seekable, and tell methods).");
    }
  }

  virtual void flush() override {
    nb::gil_scoped_acquire acquire;
    if (nb::hasattr(fileLike, "flush")) {
      fileLike.attr("flush")();
    }
  }

  virtual bool write(const char *ptr, unsigned long long numBytes) override {
    nb::gil_scoped_acquire acquire;

    for (unsigned long long i = 0; i < numBytes; i += MAX_BUFFER_SIZE) {
      unsigned long long chunkSize = std::min(numBytes - i, MAX_BUFFER_SIZE);

      int bytesWritten = nb::cast<int>(
          fileLike.attr("write")(nb::bytes((const char *)ptr, chunkSize)));

      if (bytesWritten < 0)
        return false;

      if ((unsigned long long)bytesWritten < chunkSize)
        return false;

      ptr += chunkSize;
    }

    return true;
  }
};
