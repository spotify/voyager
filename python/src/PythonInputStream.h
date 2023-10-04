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

namespace py = pybind11;

#include "PythonFileLike.h"
#include <StreamUtils.h>

bool isReadableFileLike(py::object fileLike) {
  return py::hasattr(fileLike, "read") && py::hasattr(fileLike, "seek") &&
         py::hasattr(fileLike, "tell") && py::hasattr(fileLike, "seekable");
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

  PythonInputStream(py::object fileLike) : PythonFileLike(fileLike) {
    if (!isReadableFileLike(fileLike)) {
      throw py::type_error("Expected a file-like object (with read, seek, "
                           "seekable, and tell methods).");
    }
  }

  bool isSeekable() {
    py::gil_scoped_acquire acquire;
    return fileLike.attr("seekable")().cast<bool>();
  }

  long long getTotalLength() {
    py::gil_scoped_acquire acquire;

    // TODO: Try reading a couple of Python properties that may contain the
    // total length: urllib3.response.HTTPResponse provides `length_remaining`,
    // for instance

    if (!fileLike.attr("seekable")().cast<bool>()) {
      return -1;
    }

    if (totalLength == -1) {
      long long pos = fileLike.attr("tell")().cast<long long>();
      fileLike.attr("seek")(0, 2);
      totalLength = fileLike.attr("tell")().cast<long long>();
      fileLike.attr("seek")(pos, 0);
    }

    return totalLength;
  }

  long long read(char *buffer, long long bytesToRead) {
    py::gil_scoped_acquire acquire;
    if (buffer == nullptr) {
      throw py::buffer_error(
          "C++ code attempted to read from a Python file-like object into a "
          "null destination buffer.");
    }

    if (bytesToRead < 0) {
      throw py::buffer_error("C++ code attempted to read a negative number "
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

      if (!py::isinstance<py::bytes>(readResult)) {
        std::string message = "Python file-like object was expected to return "
                              "bytes from its read(...) method, but "
                              "returned " +
                              py::str(readResult.get_type().attr("__name__"))
                                  .cast<std::string>() +
                              ".";

        if (py::hasattr(fileLike, "mode") &&
            py::str(fileLike.attr("mode")).cast<std::string>() == "r") {
          message += " (Try opening the stream in \"rb\" mode instead of "
                     "\"r\" mode if possible.)";
        }

        throw py::type_error(message);
        return 0;
      }

      py::bytes bytesObject = readResult.cast<py::bytes>();
      char *pythonBuffer = nullptr;
      py::ssize_t pythonLength = 0;

      if (PYBIND11_BYTES_AS_STRING_AND_SIZE(bytesObject.ptr(), &pythonBuffer,
                                            &pythonLength)) {
        throw py::buffer_error(
            "Internal error: failed to read bytes from bytes object!");
      }

      if (!buffer && pythonLength > 0) {
        throw py::buffer_error("Internal error: bytes pointer is null, but a "
                               "non-zero number of bytes were returned!");
      }

      if (bytesRead + pythonLength > bytesToRead) {
        throw py::buffer_error(
            "Python returned " + std::to_string(pythonLength) +
            " bytes, but only " + std::to_string(bytesToRead - bytesRead) +
            " bytes were requested.");
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
    py::gil_scoped_acquire acquire;

    if (lastReadWasSmallerThanExpected) {
      return true;
    }

    return getPosition() == getTotalLength();
  }

  long long getPosition() {
    py::gil_scoped_acquire acquire;

    return fileLike.attr("tell")().cast<long long>() - peekValue.size();
  }

  bool setPosition(long long pos) {
    py::gil_scoped_acquire acquire;

    if (fileLike.attr("seekable")().cast<bool>()) {
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