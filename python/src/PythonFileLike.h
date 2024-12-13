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
#include <string>

namespace nb = nanobind;

namespace PythonException {
// Check if there's a Python exception pending in the interpreter.
inline bool isPending() {
  nb::gil_scoped_acquire acquire;
  return PyErr_Occurred() != nullptr;
}

// If an exception is pending, raise it as a C++ exception to break the current
// control flow and result in an error being thrown in Python later.
inline void raise() {
  nb::gil_scoped_acquire acquire;

  if (PyErr_Occurred()) {
    nb::python_error existingError;
    throw existingError;
  }
}
}; // namespace PythonException

/**
 * A base class for file-like Python object wrappers.
 */
class PythonFileLike {
public:
  PythonFileLike(nb::object fileLike) : fileLike(fileLike) {}

  std::string getRepresentation() {
    nb::gil_scoped_acquire acquire;
    return nb::cast<std::string>(nb::repr(fileLike));
  }

  std::optional<std::string> getFilename() {
    // Some Python file-like objects expose a ".name" property.
    // If this object has that property, return its value;
    // otherwise return an empty optional.
    nb::gil_scoped_acquire acquire;

    if (nb::hasattr(fileLike, "name")) {
      return nb::cast<std::string>(nb::str(nb::handle(fileLike.attr("name"))));
    } else {
      return {};
    }
  }

  bool isSeekable() {
    nb::gil_scoped_acquire acquire;
    return nb::cast<bool>(fileLike.attr("seekable")());
  }

  long long getPosition() {
    nb::gil_scoped_acquire acquire;
    return nb::cast<long long>(fileLike.attr("tell")());
  }

  bool setPosition(long long pos) {
    nb::gil_scoped_acquire acquire;
    if (nb::cast<bool>(fileLike.attr("seekable")())) {
      fileLike.attr("seek")(pos);
    }

    return nb::cast<long long>(fileLike.attr("tell")()) == pos;
  }

  nb::object getFileLikeObject() { return fileLike; }

protected:
  nb::object fileLike;
};
