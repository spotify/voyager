#include <assert.h>
#include <atomic>
#include <bitset>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ratio>
#include <stdlib.h>
#include <thread>

#include "src/PythonInputStream.h"
#include "src/PythonOutputStream.h"
#include <TypedIndex.h>

namespace py = pybind11;
using namespace pybind11::literals; // needed to bring in _a literal

/**
 * Convert a PyArray (i.e.: numpy.ndarray) to a C++ NDArray.
 * This function copies the data from the PyArray into a new NDArray.
 */
template <typename T, int Dims>
NDArray<T, Dims> pyArrayToNDArray(py::array_t<T> input) {
  py::buffer_info inputInfo = input.request();

  if (inputInfo.ndim != Dims) {
    throw std::domain_error("Input array was expected to have rank " +
                            std::to_string(Dims) + ", but had rank " +
                            std::to_string(inputInfo.ndim) + ".");
  }

  std::array<int, Dims> shape;
  for (int i = 0; i < Dims; i++) {
    shape[i] = inputInfo.shape[i];
  }

  NDArray<T, Dims> output = NDArray<T, Dims>(shape);

  T *inputPtr = static_cast<T *>(inputInfo.ptr);
  {
    py::gil_scoped_release release;
    std::copy(inputPtr, inputPtr + output.data.size(), output.data.begin());
  }

  return output;
};

/**
 * Convert a C++ NDArray into a PyArray (i.e.: numpy.ndarray).
 * This function copies the data, but may not have to if NDArray were
 * refactored.
 */
template <typename T, int Dims>
py::array_t<T> ndArrayToPyArray(NDArray<T, Dims> input) {
  py::array_t<T> output(input.shape);
  T *outputPtr = static_cast<T *>(const_cast<T *>(output.data()));

  {
    py::gil_scoped_release release;
    std::copy(input.data.begin(), input.data.end(), outputPtr);
  }

  return output;
};

/**
 * Convert a PyArray (i.e.: numpy.ndarray) to a C++ std::vector.
 * This function copies the data from the PyArray into a new vector.
 */
template <typename T> std::vector<T> pyArrayToVector(py::array_t<T> input) {
  py::buffer_info inputInfo = input.request();

  if (inputInfo.ndim != 1) {
    throw std::domain_error(
        "Input array was expected to have one dimension, but had " +
        std::to_string(inputInfo.ndim) + " dimensions.");
  }

  std::vector<T> output(inputInfo.shape[0]);

  T *inputPtr = static_cast<T *>(inputInfo.ptr);
  {
    py::gil_scoped_release release;
    std::copy(inputPtr, inputPtr + output.size(), output.begin());
  }

  return output;
};

/**
 * Convert a C++ std::vector into a PyArray (i.e.: numpy.ndarray).
 * This function copies the data, but may not have to.
 */
template <typename T> py::array_t<T> vectorToPyArray(std::vector<T> input) {
  py::array_t<T> output({(long)input.size()});
  T *outputPtr = static_cast<T *>(const_cast<T *>(output.data()));

  {
    py::gil_scoped_release release;
    std::copy(input.begin(), input.end(), outputPtr);
  }

  return output;
};

/**
 * A lazy Python list-like object that allows for iteration through
 * a huge list without having to materialize the whole list and copy
 * its contents.
 */
class SetView {
public:
  SetView(const std::unordered_map<hnswlib::labeltype, hnswlib::tableint> &map)
      : map(map) {}
  std::unordered_map<hnswlib::labeltype, hnswlib::tableint> const &map;
};

inline void init_SetView(py::module &m) {
  py::class_<SetView>(m, "SetView",
                      "A read-only set-like object containing 64-bit integers "
                      "that is backed by the "
                      "unordered keys of a C++ map.")
      .def("__repr__",
           [](SetView &self) {
             std::ostringstream ss;
             ss << "<voyager.SetView";
             ss << " num_elements=" << self.map.size();
             ss << " at " << &self;
             ss << ">";
             return ss.str();
           })
      .def("__len__", [](SetView &self) { return self.map.size(); })
      .def("__iter__",
           [](SetView &self) {
             // Python iterates much faster through a list of longs
             // than when jumping back and forth between Python and C++
             // every time next(iter(...)) is called.
             std::vector<hnswlib::labeltype> ids;
             {
               py::gil_scoped_release release;
               ids.reserve(self.map.size());
               for (auto const &kv : self.map) {
                 ids.push_back(kv.first);
               }
             }

             return py::cast(ids).attr("__iter__")();
           })
      .def(
          "__contains__",
          [](SetView &self, hnswlib::labeltype element) {
            return self.map.find(element) != self.map.end();
          },
          py::arg("id"))
      .def(
          "__contains__", [](SetView &, const py::object &) { return false; },
          py::arg("id"));
}

template <typename dist_t, typename data_t,
          typename scalefactor = std::ratio<1, 1>>
inline void register_index_class(py::module &m, std::string className) {
  auto klass =
      py::class_<TypedIndex<dist_t, data_t, scalefactor>, Index,
                 std::shared_ptr<TypedIndex<dist_t, data_t, scalefactor>>>(
          m, className.c_str());

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Index Construction and Indexing
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  klass.def(py::init<const SpaceType, const int, const size_t, const size_t,
                     const size_t, const size_t>(),
            py::arg("space"), py::arg("num_dimensions"), py::arg("M") = 16,
            py::arg("ef_construction") = 200, py::arg("random_seed") = 1,
            py::arg("max_elements") = 1, "Create a new, empty index.");

  klass.def(
      "add_item",
      [](TypedIndex<dist_t, data_t, scalefactor> &index,
         py::array_t<float> vector, std::optional<size_t> _id) {
        auto stdArray = pyArrayToVector<float>(vector);

        py::gil_scoped_release release;
        index.addItem(stdArray, _id);
      },
      py::arg("vector"), py::arg("id") = py::none());

  klass.def(
      "add_items",
      [](TypedIndex<dist_t, data_t, scalefactor> &index,
         py::array_t<float> vectors, std::optional<std::vector<size_t>> _ids,
         int num_threads) {
        std::vector<size_t> empty;
        auto ndArray = pyArrayToNDArray<float, 2>(vectors);

        py::gil_scoped_release release;
        index.addItems(ndArray, (_ids ? *_ids : empty), num_threads);
      },
      py::arg("vectors"), py::arg("ids") = py::none(),
      py::arg("num_threads") = -1);

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Querying
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  klass.def(
      "query",
      [](TypedIndex<dist_t, data_t, scalefactor> &index,
         py::array_t<float> input, size_t k = 1, int num_threads = -1,
         long queryEf = -1) {
        int inputNDim = input.request().ndim;
        switch (inputNDim) {
        case 1: {
          auto idsAndDistances =
              index.query(pyArrayToVector<float>(input), k, queryEf);
          std::tuple<py::array_t<hnswlib::labeltype>, py::array_t<float>>
              output = {vectorToPyArray<hnswlib::labeltype>(
                            std::get<0>(idsAndDistances)),
                        vectorToPyArray<float>(std::get<1>(idsAndDistances))};
          return output;
        }
        case 2: {
          auto idsAndDistances = index.query(pyArrayToNDArray<float, 2>(input),
                                             k, num_threads, queryEf);
          std::tuple<py::array_t<hnswlib::labeltype>, py::array_t<float>>
              output = {
                  ndArrayToPyArray<hnswlib::labeltype, 2>(
                      std::get<0>(idsAndDistances)),
                  ndArrayToPyArray<float, 2>(std::get<1>(idsAndDistances))};
          return output;
        }
        default:
          throw std::domain_error(
              "query(...) expected one- or two-dimensional input data (either "
              "a single query vector or multiple query vectors) but got " +
              std::to_string(inputNDim) + " dimensions.");
        }
      },
      py::arg("vectors"), py::arg("k") = 1, py::arg("num_threads") = -1,
      py::arg("query_ef") = -1);

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Property Methods
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  klass.def_property_readonly(
      "space", &TypedIndex<dist_t, data_t, scalefactor>::getSpace);

  klass.def_property_readonly(
      "num_dimensions",
      &TypedIndex<dist_t, data_t, scalefactor>::getNumDimensions);

  klass.def_property_readonly("M",
                              &TypedIndex<dist_t, data_t, scalefactor>::getM);

  klass.def_property_readonly(
      "ef_construction",
      &TypedIndex<dist_t, data_t, scalefactor>::getEfConstruction);

  klass.def_property_readonly(
      "max_elements", &TypedIndex<dist_t, data_t, scalefactor>::getMaxElements);

  // TODO: Add getStorageDataType

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Index Accessor Methods
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  klass.def_property_readonly(
      "num_elements", &TypedIndex<dist_t, data_t, scalefactor>::getNumElements);

  klass.def(
      "get_vector",
      [](TypedIndex<dist_t, data_t, scalefactor> &index, size_t _id) {
        return ndArrayToPyArray<float, 1>(NDArray<float, 1>(
            index.getVector(_id), {(int)index.getNumDimensions()}));
      },
      py::arg("id"));

  klass.def(
      "get_vectors",
      [](TypedIndex<dist_t, data_t, scalefactor> &index,
         std::vector<size_t> _ids) {
        return ndArrayToPyArray<float, 2>(index.getVectors(_ids));
      },
      py::arg("ids"));

  klass.def_property_readonly(
      "ids",
      [](TypedIndex<dist_t, data_t, scalefactor> &index) {
        return std::make_unique<SetView>(index.getIDsMap());
      },
      "A set(-like object) containing the integer IDs stored as "
      "'keys' in this index. May be extremely long.");

  klass.def(
      "get_distance",
      [](TypedIndex<dist_t, data_t, scalefactor> &index, std::vector<float> a,
         std::vector<float> b) { return index.getDistance(a, b); },
      "Get the distance between two provided vectors. The vectors must share "
      "the dimensionality of the index.",
      py::arg("a"), py::arg("b"));

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Index Modifier Methods/Attributes
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  klass.def_property("ef", &TypedIndex<dist_t, data_t, scalefactor>::getEF,
                     &TypedIndex<dist_t, data_t, scalefactor>::setEF);

  klass.def("mark_deleted",
            &TypedIndex<dist_t, data_t, scalefactor>::markDeleted,
            py::arg("label"));

  klass.def("unmark_deleted",
            &TypedIndex<dist_t, data_t, scalefactor>::unmarkDeleted,
            py::arg("label"));

  klass.def("resize_index",
            &TypedIndex<dist_t, data_t, scalefactor>::resizeIndex,
            py::arg("new_size"));

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Save Index
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  klass.def(
      "save_index",
      [](TypedIndex<dist_t, data_t, scalefactor> &index, std::string filePath) {
        py::gil_scoped_release release;
        index.saveIndex(filePath);
      },
      py::arg("output_path"), "Save this index to the provided file path.");

  klass.def(
      "save_index",
      [](TypedIndex<dist_t, data_t, scalefactor> &index, py::object filelike) {
        auto outputStream = std::make_shared<PythonOutputStream>(filelike);

        py::gil_scoped_release release;
        index.saveIndex(outputStream);
      },
      py::arg("file_like"),
      "Save this index to the provided file-like object.");

  klass.def(
      "as_bytes",
      [](TypedIndex<dist_t, data_t, scalefactor> &index) {
        auto outputStream = std::make_shared<MemoryOutputStream>();
        {
          py::gil_scoped_release release;
          index.saveIndex(outputStream);
        }

        return py::bytes(outputStream->getValue());
      },
      "Returns the byte contents of this index. This may be extremely large "
      "(many GB) if the index is sufficiently large. To save to disk without "
      "allocating this entire bytestring, use ``save_index``.");

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Python Builtin Supports
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  klass.attr("__bytes__") = klass.attr("as_bytes");
  klass.def("__repr__", [className](const Index &index) {
    return "<voyager." + className + " space=" + index.getSpaceName() +
           " num_dimensions=" + std::to_string(index.getNumDimensions()) +
           " storage_data_type=" + index.getStorageDataTypeName() + ">";
  });
};

PYBIND11_MODULE(voyager, m) {
  init_SetView(m);

  py::enum_<SpaceType>(m, "Space", "The type of space to use for searching.")
      .value("Euclidean", SpaceType::Euclidean)
      .value("Cosine", SpaceType::Cosine)
      .value("InnerProduct", SpaceType::InnerProduct)
      .export_values();

  py::enum_<StorageDataType>(
      m, "StorageDataType",
      "The datatype used to store vectors in memory and on-disk.")
      .value("Float8", StorageDataType::Float8,
             "8-bit floating point. All values must be within [-1, 1].")
      .value("Float32", StorageDataType::Float32,
             "32-bit floating point (default).")
      .value("E4M3", StorageDataType::E4M3,
             "8-bit floating point with a range of [-448, 448], inspired by "
             "the paper \"FP8 Formats for Deep Learning\" by Micikevicius et "
             "al. (arXiv:2209.05433)")
      .export_values();

  py::class_<E4M3>(
      m, "E4M3T",
      "An 8-bit floating point data type with reduced precision and range.")
      .def(py::init([](float input) {
             E4M3 v(input);
             return v;
           }),
           "Create an E4M3 number given a floating-point value. If out of "
           "range, the value will be clipped.")
      .def(py::init([](int sign, int exponent, int mantissa) {
             E4M3 v(sign, exponent, mantissa);
             return v;
           }),
           "Create an E4M3 number given a sign, exponent, and mantissa. If out "
           "of range, the values will be clipped.")
      .def_static(
          "from_char",
          [](int c) {
            if (c > 255 || c < 0)
              throw std::range_error(
                  "Expected input to from_char to be on [0, 255]!");
            E4M3 v(static_cast<uint8_t>(c));
            return v;
          },
          "Create an E4M3 number given a raw 8-bit value.", py::arg("value"))
      .def(
          "__float__", [](E4M3 &self) { return (float)self; },
          "Cast the given E4M3 number to a float.")
      .def("__repr__",
           [](E4M3 &self) {
             std::ostringstream ss;
             ss << "<voyager.E4M3";
             ss << " sign=" << (int)self.sign;
             ss << " exponent=" << (int)self.effectiveExponent() << " ("
                << std::bitset<4>(self.exponent) << ")";
             ss << " mantissa=" << self.effectiveMantissa() << " ("
                << std::bitset<3>(self.mantissa) << ")";
             ss << " float=" << ((float)self);
             ss << " at " << &self;
             ss << ">";
             return ss.str();
           })
      .def_property_readonly(
          "sign", [](E4M3 &self) { return self.sign; },
          "The sign bit from this E4M3 number.")
      .def_property_readonly(
          "exponent", [](E4M3 &self) { return self.exponent; },
          "The exponent bit from this E4M3 number.")
      .def_property_readonly(
          "mantissa", [](E4M3 &self) { return self.mantissa; },
          "The mantissa bit from this E4M3 number.")
      .def_property_readonly(
          "size", [](E4M3 &self) { return sizeof(self); },
          "The number of bytes used to represent this (C++) instance in "
          "memory.");

  auto index = py::class_<Index, std::shared_ptr<Index>>(m, "Index");

  register_index_class<float, float>(m, "FloatIndex");
  // An int8 index that stores its values as 8-bit integers, assumes all
  // input/output data is float on [-1, 1], and returns floating-point
  // distances.
  register_index_class<float, int8_t, std::ratio<1, 127>>(m, "Float8Index");

  // An 8-bit floating-point index class that has even more reduced precision
  // over Float8, but allows values on the range [-448, 448]. Inspired by:
  // https://arxiv.org/pdf/2209.05433.pdf
  register_index_class<float, E4M3>(m, "E4M3Index");

  index.def_static(
      "__new__",
      [](const py::object *, const SpaceType space, const int num_dimensions,
         const size_t M, const size_t ef_construction, const size_t random_seed,
         const size_t max_elements) {
        py::gil_scoped_release release;
        return std::make_shared<TypedIndex<float>>(space, num_dimensions, M,
                                                   ef_construction, random_seed,
                                                   max_elements);
      },
      py::arg("cls"), py::arg("space"), py::arg("num_dimensions"),
      py::arg("M") = 12, py::arg("ef_construction") = 200,
      py::arg("random_seed") = 1, py::arg("max_elements") = 1);

  index.def_static(
      "__new__",
      [](const py::object *, const SpaceType space, const int num_dimensions,
         const size_t M, const size_t ef_construction, const size_t random_seed,
         const size_t max_elements,
         const StorageDataType storageDataType) -> std::shared_ptr<Index> {
        py::gil_scoped_release release;
        switch (storageDataType) {
        case StorageDataType::E4M3:
          return std::make_shared<TypedIndex<float, E4M3>>(
              space, num_dimensions, M, ef_construction, random_seed,
              max_elements);
        case StorageDataType::Float8:
          return std::make_shared<
              TypedIndex<float, int8_t, std::ratio<1, 127>>>(
              space, num_dimensions, M, ef_construction, random_seed,
              max_elements);
        case StorageDataType::Float32:
          return std::make_shared<TypedIndex<float>>(space, num_dimensions, M,
                                                     ef_construction,
                                                     random_seed, max_elements);
        default:
          throw std::runtime_error("Unknown storage data type received!");
        }
      },
      py::arg("cls"), py::arg("space"), py::arg("num_dimensions"),
      py::arg("M") = 12, py::arg("ef_construction") = 200,
      py::arg("random_seed") = 1, py::arg("max_elements") = 1,
      py::arg("storage_data_type") = StorageDataType::Float32);

  index.def_static(
      "load",
      [](const std::string filename, const SpaceType space,
         const int num_dimensions,
         const StorageDataType storageDataType) -> std::shared_ptr<Index> {
        py::gil_scoped_release release;

        switch (storageDataType) {
        case StorageDataType::E4M3:
          return std::make_shared<TypedIndex<float, E4M3>>(
              std::make_shared<FileInputStream>(filename), space,
              num_dimensions);
        case StorageDataType::Float8:
          return std::make_shared<
              TypedIndex<float, int8_t, std::ratio<1, 127>>>(
              std::make_shared<FileInputStream>(filename), space,
              num_dimensions);
        case StorageDataType::Float32:
          return std::make_shared<TypedIndex<float>>(
              std::make_shared<FileInputStream>(filename), space,
              num_dimensions);
        default:
          throw std::runtime_error("Unknown storage data type received!");
        }
      },
      py::arg("filename"), py::arg("space"), py::arg("num_dimensions"),
      py::arg("storage_data_type") = StorageDataType::Float32,
      "Load an index from a .voy or .hnsw file, provided its filename.");

  index.def_static(
      "load",
      [](const py::object filelike, const SpaceType space,
         const int num_dimensions,
         const StorageDataType storageDataType) -> std::shared_ptr<Index> {
        if (!isReadableFileLike(filelike)) {
          throw py::type_error(
              "Expected either a filename or a file-like object (with "
              "read, seek, seekable, and tell methods), but received: " +
              filelike.attr("__repr__")().cast<std::string>());
        }

        auto inputStream = std::make_shared<PythonInputStream>(filelike);
        py::gil_scoped_release release;

        switch (storageDataType) {
        case StorageDataType::E4M3:
          return std::make_shared<TypedIndex<float, E4M3>>(inputStream, space,
                                                           num_dimensions);
        case StorageDataType::Float8:
          return std::make_shared<
              TypedIndex<float, int8_t, std::ratio<1, 127>>>(inputStream, space,
                                                             num_dimensions);
        case StorageDataType::Float32:
          return std::make_shared<TypedIndex<float>>(inputStream, space,
                                                     num_dimensions);
        default:
          throw std::runtime_error("Unknown storage data type received!");
        }
      },
      py::arg("file_like"), py::arg("space"), py::arg("num_dimensions"),
      py::arg("storage_data_type") = StorageDataType::Float32,
      "Load an index from a file-like object. The provided object must have "
      "``read``, ``seek``, ``tell``, and ``seekable`` methods, and must "
      "return binary data (i.e.: ``open(..., \"w\")`` or ``io.BinaryIO``, "
      "etc.).");

  index.def_static(
      "load_from_subprocess",
      [](const std::string subprocessCommand, const SpaceType space,
         const int num_dimensions,
         const StorageDataType storageDataType) -> std::shared_ptr<Index> {
        py::gil_scoped_release release;

        switch (storageDataType) {
        case StorageDataType::E4M3:
          return std::make_shared<TypedIndex<float, E4M3>>(
              std::make_shared<SubprocessInputStream>(subprocessCommand), space,
              num_dimensions);
        case StorageDataType::Float8:
          return std::make_shared<
              TypedIndex<float, int8_t, std::ratio<1, 127>>>(
              std::make_shared<SubprocessInputStream>(subprocessCommand), space,
              num_dimensions);
        case StorageDataType::Float32:
          return std::make_shared<TypedIndex<float>>(
              std::make_shared<SubprocessInputStream>(subprocessCommand), space,
              num_dimensions);
        default:
          throw std::runtime_error("Unknown storage data type received!");
        }
      },
      py::arg("subprocess_command"), py::arg("space"),
      py::arg("num_dimensions"),
      py::arg("storage_data_type") = StorageDataType::Float32,
      "Load an index from the standard output stream of a subprocess "
      "command. This can be used to load an index from a remote filesystem "
      "(i.e.: S3, GCS, etc) extremely quickly, using all of the available "
      "threads/cores on a machine, without downloading to disk first.");
}
