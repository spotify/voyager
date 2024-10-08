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

bool isReadableFileLike(nb::object fileLike) {
  return nb::hasattr(fileLike, "read") && nb::hasattr(fileLike, "seek") &&
         nb::hasattr(fileLike, "tell") && nb::hasattr(fileLike, "seekable");
}

/**
 * An input stream that fetches its
 * data from a provided Python file-like object.
 */
class PythonInputStream : public InputStream, PythonFileLike {
public:
  // This input stream stores a temporary buffer to copy between Python and C++;
  // if we don't set a maximum buffer size here, the C++ side could read
  // hundreds of GB at once, which would allocate 2x that amount.
  static constexpr long long MAX_BUFFER_SIZE = 1024 * 1024 * 100;

  PythonInputStream(nb::object fileLike) : PythonFileLike(fileLike) {
    if (!isReadableFileLike(fileLike)) {
      throw nb::type_error("Expected a file-like object (with read, seek, "
                           "seekable, and tell methods).");
    }
  }

  bool isSeekable() {
    nb::gil_scoped_acquire acquire;
    return nb::cast<bool>(fileLike.attr("seekable")());
  }

  long long getTotalLength() {
    nb::gil_scoped_acquire acquire;

    // TODO: Try reading a couple of Python properties that may contain the
    // total length: urllib3.response.HTTPResponse provides `length_remaining`,
    // for instance

    if (!nb::cast<bool>(fileLike.attr("seekable")())) {
      return -1;
    }

    if (totalLength == -1) {
      long long pos = nb::cast<long long>(fileLike.attr("tell")());
      fileLike.attr("seek")(0, 2);
      totalLength = nb::cast<long long>(fileLike.attr("tell")());
      fileLike.attr("seek")(pos, 0);
    }

    return totalLength;
  }

  long long read(char *buffer, long long bytesToRead) {
    nb::gil_scoped_acquire acquire;
    if (buffer == nullptr) {
      throw nb::buffer_error(
          "C++ code attempted to read from a Python file-like object into a "
          "null destination buffer.");
    }

    if (bytesToRead < 0) {
      throw nb::buffer_error("C++ code attempted to read a negative number "
                             "of bytes from a Python file-like object.");
    }

    long long bytesRead = 0;

    if (peekValue.size()) {
      long long bytesToCopy =
          std::min(bytesToRead, (long long)peekValue.size());
      std::memcpy(buffer, peekValue.data(), bytesToCopy);
      for (int i = 0; i < bytesToCopy; i++)
        peekValue.erase(peekValue.begin());
      bytesRead += bytesToCopy;
      buffer += bytesToCopy;
    }

    while (bytesRead < bytesToRead) {
      auto readResult = fileLike.attr("read")(
          std::min(MAX_BUFFER_SIZE, bytesToRead - bytesRead));

      if (!nb::isinstance<nb::bytes>(readResult)) {
        std::string message =
            "Python file-like object was expected to return "
            "bytes from its read(...) method, but "
            "returned " +
            nb::cast<std::string>(nb::str(readResult.type().attr("__name__"))) +
            ".";

        if (nb::hasattr(fileLike, "mode") &&
            nb::cast<std::string>(nb::str(fileLike.attr("mode"))) == "r") {
          message += " (Try opening the stream in \"rb\" mode instead of "
                     "\"r\" mode if possible.)";
        }

        throw nb::type_error(message.c_str());
        return 0;
      }

      nb::bytes bytesObject = nb::cast<nb::bytes>(readResult);
      const char *pythonBuffer = bytesObject.c_str();
      nb::ssize_t pythonLength = bytesObject.size();

      if (!buffer && pythonLength > 0) {
        throw nb::buffer_error("Internal error: bytes pointer is null, but a "
                               "non-zero number of bytes were returned!");
      }

      if (bytesRead + pythonLength > bytesToRead) {
        throw nb::buffer_error(
            ("Python returned " + std::to_string(pythonLength) +
             " bytes, but only " + std::to_string(bytesToRead - bytesRead) +
             " bytes were requested.")
                .c_str());
      }

      if (buffer && pythonLength > 0) {
        std::memcpy(buffer, pythonBuffer, pythonLength);
        bytesRead += pythonLength;
        buffer += pythonLength;
      } else {
        break;
      }
    }

    lastReadWasSmallerThanExpected = bytesToRead > bytesRead;
    return bytesRead;
  }

  bool isExhausted() {
    nb::gil_scoped_acquire acquire;

    if (lastReadWasSmallerThanExpected) {
      return true;
    }

    return getPosition() == getTotalLength();
  }

  long long getPosition() {
    nb::gil_scoped_acquire acquire;

    return nb::cast<long long>(fileLike.attr("tell")()) - peekValue.size();
  }

  bool setPosition(long long pos) {
    nb::gil_scoped_acquire acquire;

    if (nb::cast<bool>(fileLike.attr("seekable")())) {
      fileLike.attr("seek")(pos);
    }

    return getPosition() == pos;
  }

  uint32_t peek() {
    uint32_t result = 0;
    long long lastPosition = getPosition();
    if (read((char *)&result, sizeof(result)) == sizeof(result)) {
      char *resultAsCharacters = (char *)&result;
      peekValue.push_back(resultAsCharacters[0]);
      peekValue.push_back(resultAsCharacters[1]);
      peekValue.push_back(resultAsCharacters[2]);
      peekValue.push_back(resultAsCharacters[3]);
      return result;
    } else {
      throw std::runtime_error("Failed to peek " +
                               std::to_string(sizeof(result)) +
                               " bytes from file-like object at index " +
                               std::to_string(lastPosition) + ".");
    }
  }

private:
  long long totalLength = -1;
  std::vector<char> peekValue;
  bool lastReadWasSmallerThanExpected = false;
};
