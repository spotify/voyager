/*
 * PythonFileLike
 * Copyright 2022 Spotify AB
 */

#pragma once

#include <mutex>
#include <optional>

namespace py = pybind11;

namespace PythonException {
// Check if there's a Python exception pending in the interpreter.
inline bool isPending() {
  py::gil_scoped_acquire acquire;
  return PyErr_Occurred() != nullptr;
}

// If an exception is pending, raise it as a C++ exception to break the current
// control flow and result in an error being thrown in Python later.
inline void raise() {
  py::gil_scoped_acquire acquire;

  if (PyErr_Occurred()) {
    py::error_already_set existingError;
    throw existingError;
  }
}
}; // namespace PythonException

/**
 * A base class for file-like Python object wrappers.
 */
class PythonFileLike {
public:
  PythonFileLike(py::object fileLike) : fileLike(fileLike) {}

  std::string getRepresentation() {
    py::gil_scoped_acquire acquire;
    return py::repr(fileLike).cast<std::string>();
  }

  std::optional<std::string> getFilename() {
    // Some Python file-like objects expose a ".name" property.
    // If this object has that property, return its value;
    // otherwise return an empty optional.
    py::gil_scoped_acquire acquire;

    if (py::hasattr(fileLike, "name")) {
      return py::str(fileLike.attr("name")).cast<std::string>();
    } else {
      return {};
    }
  }

  bool isSeekable() {
    py::gil_scoped_acquire acquire;
    return fileLike.attr("seekable")().cast<bool>();
  }

  long long getPosition() {
    py::gil_scoped_acquire acquire;
    return fileLike.attr("tell")().cast<long long>();
  }

  bool setPosition(long long pos) {
    py::gil_scoped_acquire acquire;
    if (fileLike.attr("seekable")().cast<bool>()) {
      fileLike.attr("seek")(pos);
    }

    return fileLike.attr("tell")().cast<long long>() == pos;
  }

  py::object getFileLikeObject() { return fileLike; }

protected:
  py::object fileLike;
};