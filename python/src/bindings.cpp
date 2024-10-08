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

#include <assert.h>
#include <atomic>
#include <bitset>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <ratio>
#include <stdlib.h>
#include <thread>

#include "../../cpp/src/TypedIndex.h"
#include "PythonInputStream.h"
#include "PythonOutputStream.h"

namespace nb = nanobind;
using namespace nanobind::literals; // needed to bring in _a literal

/**
 * Convert a PyArray (i.e.: numpy.ndarray) to a C++ NDArray.
 * This function copies the data from the PyArray into a new NDArray.
 */
template <typename T, int Dims>
NDArray<T, Dims> pyArrayToNDArray(nb::ndarray<T> &input) {
  if (input.ndim() != Dims) {
    throw std::domain_error("Input array was expected to have rank " +
                            std::to_string(Dims) + ", but had rank " +
                            std::to_string(input.ndim()) + ".");
  }

  std::array<int, Dims> shape;
  for (int i = 0; i < Dims; i++) {
    shape[i] = input.shape(i);
  }

  NDArray<T, Dims> output = NDArray<T, Dims>(shape);

  const T *inputPtr = static_cast<const T *>(input.data());
  {
    nb::gil_scoped_release release;
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
nb::ndarray<T, nb::numpy> ndArrayToPyArray(NDArray<T, Dims> input) {
  T *outputPtr = new T[input.data.size()];
  std::copy(input.data.begin(), input.data.end(), outputPtr);
  nb::capsule owner(outputPtr, [](void *p) noexcept { delete[] (float *)p; });
  size_t shape[Dims];
  for (int i = 0; i < Dims; i++) {
    shape[i] = input.shape[i];
  }
  return nb::ndarray<T, nb::numpy>(outputPtr, Dims, shape, owner);
};

/**
 * Convert a PyArray (i.e.: numpy.ndarray) to a C++ std::vector.
 * This function copies the data from the PyArray into a new vector.
 */
template <typename T> std::vector<T> pyArrayToVector(nb::ndarray<T> &input) {
  if (input.ndim() != 1) {
    throw std::domain_error(
        "Input array was expected to have one dimension, but had " +
        std::to_string(input.ndim()) + " dimensions.");
  }

  std::vector<T> output(input.shape(0));

  const T *inputPtr = static_cast<const T *>(input.data());
  {
    nb::gil_scoped_release release;
    std::copy(inputPtr, inputPtr + output.size(), output.begin());
  }

  return output;
};

/**
 * Convert a C++ std::vector into a PyArray (i.e.: numpy.ndarray).
 * This function copies the data, but may not have to.
 */
template <typename T>
nb::ndarray<T, nb::numpy> vectorToPyArray(std::vector<T> input) {
  T *data = new T[input.size()];
  std::copy(input.begin(), input.end(), data);
  nb::capsule owner(data, [](void *p) noexcept { delete[] (float *)p; });
  return nb::ndarray<T, nb::numpy>(data, {input.size()}, owner);
};

/**
 * A lazy Python list-like object that allows for iteration through
 * a huge list without having to materialize the whole list and copy
 * its contents.
 */
class LabelSetView {
public:
  LabelSetView(
      const std::unordered_map<hnswlib::labeltype, hnswlib::tableint> &map)
      : map(map) {}
  std::unordered_map<hnswlib::labeltype, hnswlib::tableint> const &map;
};

inline void init_LabelSetView(nb::module_ &m) {
  nb::class_<LabelSetView>(
      m, "LabelSetView",
      "A read-only set-like object containing 64-bit integers. Use this object "
      "like a regular Python :py:class:`set` object, by either iterating "
      "through it, or checking for membership with the ``in`` operator.")
      .def("__repr__",
           [](LabelSetView &self) {
             std::ostringstream ss;
             ss << "<voyager.LabelSetView";
             ss << " num_elements=" << self.map.size();
             ss << " at " << &self;
             ss << ">";
             return ss.str();
           })
      .def("__len__", [](LabelSetView &self) { return self.map.size(); })
      .def("__iter__",
           [](LabelSetView &self) {
             // Python iterates much faster through a list of longs
             // than when jumping back and forth between Python and C++
             // every time next(iter(...)) is called.
             std::vector<hnswlib::labeltype> ids;
             {
               nb::gil_scoped_release release;
               ids.reserve(self.map.size());
               for (auto const &kv : self.map) {
                 ids.push_back(kv.first);
               }
             }

             return nb::cast(ids).attr("__iter__")();
           })
      .def(
          "__contains__",
          [](LabelSetView &self, hnswlib::labeltype element) {
            return self.map.find(element) != self.map.end();
          },
          nb::arg("id"))
      .def(
          "__contains__",
          [](LabelSetView &, const nb::object &) { return false; },
          nb::arg("id"));
}

template <typename dist_t, typename data_t,
          typename scalefactor = std::ratio<1, 1>>
inline void register_index_class(nb::module_ &m, std::string className,
                                 std::string docstring) {
  nb::class_<TypedIndex<dist_t, data_t, scalefactor>, Index>(
      m, className.c_str(), docstring.c_str())
      .def(
          "__init__",
          [](
              // TypedIndex<dist_t, data_t, scalefactor> *self,
              const nb::object *self, const SpaceType space,
              const int num_dimensions, const size_t M,
              const size_t ef_construction, const size_t random_seed,
              const size_t max_elements,
              const StorageDataType storageDataType) {
            // new (self) TypedIndex<dist_t, data_t, scalefactor>(
            //     space, num_dimensions, M, ef_construction, random_seed,
            //     max_elements);
          },
          nb::arg("space"), nb::arg("num_dimensions"), nb::arg("M") = 16,
          nb::arg("ef_construction") = 200, nb::arg("random_seed") = 1,
          nb::arg("max_elements") = 1,
          nb::arg("storage_data_type") = StorageDataType::Float32,
          "Create a new, empty index.")
      .def("__repr__", [className](const Index &index) {
        return "<voyager." + className + " space=" + index.getSpaceName() +
               " num_dimensions=" + std::to_string(index.getNumDimensions()) +
               " storage_data_type=" + index.getStorageDataTypeName() + ">";
      });
};

NB_MODULE(voyager_ext, m) {
  nb::exception<RecallError>(m, "RecallError");

  m.attr("version") = nb::make_tuple(2, 0, 10);

  init_LabelSetView(m);

  nb::enum_<SpaceType>(
      m, "Space", "The method used to calculate the distance between vectors.")
      .value("Euclidean", SpaceType::Euclidean,
             "Euclidean distance; also known as L2 distance. The square root "
             "of the sum of the squared differences between each element of "
             "each vector.")
      .value("Cosine", SpaceType::Cosine,
             "Cosine distance; also known as normalized inner product.")
      .value("InnerProduct", SpaceType::InnerProduct, "Inner product distance.")
      .export_values();

  nb::enum_<StorageDataType>(m, "StorageDataType",
                             R"(
The data type used to store vectors in memory and on-disk.

The :py:class:`StorageDataType` used for an :py:class:`Index` directly determines
its memory usage, disk space usage, and recall. Both :py:class:`Float8` and
:py:class:`E4M3` data types use 8 bits (1 byte) per dimension per vector, reducing
memory usage and index size by a factor of 4 compared to :py:class:`Float32`.
)")
      .value("Float8", StorageDataType::Float8,
             "8-bit fixed-point decimal values. All values must be within [-1, "
             "1.00787402].")
      .value("Float32", StorageDataType::Float32,
             "32-bit floating point (default).")
      .value("E4M3", StorageDataType::E4M3,
             "8-bit floating point with a range of [-448, 448], from "
             "the paper \"FP8 Formats for Deep Learning\" by Micikevicius et "
             "al.\n\n.. warning::\n    Using E4M3 with the Cosine "
             ":py:class:`Space` may cause some queries to return "
             "negative distances due to the reduced floating-point precision. "
             "While confusing, these negative distances still result in a "
             "correct ordering between results.")
      .export_values();

  nb::class_<E4M3>(
      m, "E4M3T",
      "An 8-bit floating point data type with reduced precision and range. "
      "This class wraps a C++ struct and should probably not be used directly.")
      .def(nb::init<float>(),
           "Create an E4M3 number given a floating-point value. If out of "
           "range, the value will be clipped.",
           nb::arg("value"))
      .def(nb::init<uint8_t, uint8_t, uint8_t>(),
           "Create an E4M3 number given a sign, exponent, and mantissa. If out "
           "of range, the values will be clipped.",
           nb::arg("sign"), nb::arg("exponent"), nb::arg("mantissa"))
      .def_static(
          "from_char",
          [](int c) {
            if (c > 255 || c < 0)
              throw std::range_error(
                  "Expected input to from_char to be on [0, 255]!");
            E4M3 v(static_cast<uint8_t>(c));
            return v;
          },
          "Create an E4M3 number given a raw 8-bit value.", nb::arg("value"))
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
      .def_prop_ro(
          "sign", [](E4M3 &self) { return self.sign; },
          "The sign bit from this E4M3 number. Will be ``1`` if the number is "
          "negative, ``0`` otherwise.")
      .def_prop_ro(
          "exponent", [](E4M3 &self) { return self.effectiveExponent(); },
          "The effective exponent of this E4M3 number, expressed as "
          "an integer.")
      .def_prop_ro(
          "raw_exponent", [](E4M3 &self) { return self.exponent; },
          "The raw value of the exponent part of this E4M3 number, expressed "
          "as an integer.")
      .def_prop_ro(
          "mantissa", [](E4M3 &self) { return self.effectiveMantissa(); },
          "The effective mantissa (non-exponent part) of this E4M3 number, "
          "expressed as an integer.")
      .def_prop_ro(
          "raw_mantissa", [](E4M3 &self) { return self.mantissa; },
          "The raw value of the mantissa (non-exponent part) of this E4M3 "
          "number, expressed as a floating point value.")
      .def_prop_ro(
          "size", [](E4M3 &self) { return sizeof(self); },
          "The number of bytes used to represent this (C++) instance in "
          "memory.");

  auto index = nb::class_<Index>(m, "Index",
                                 R"(
A nearest-neighbor search index containing vector data (i.e. lists of 
floating-point values, each list labeled with a single integer ID).

Think of a Voyager :py:class:`Index` as a ``Dict[int, List[float]]``
(a dictionary mapping integers to lists of floats), where a
:py:meth:`query` method allows finding the *k* nearest ``int`` keys
to a provided ``List[float]`` query vector.

.. warning::
    Voyager is an **approximate** nearest neighbor index package, which means
    that each call to the :py:meth:`query` method may return results that
    are *approximately* correct. The metric used to gauge the accuracy
    of a Voyager :py:meth:`query` is called
    `recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_. Recall
    measures the percentage of correct results returned per
    :py:meth:`query`.
    
    Various parameters documented below (like ``M``, ``ef_construction``,
    and ``ef``) may affect the recall of queries. Usually, increasing recall
    requires either increasing query latency, increasing memory usage,
    increasing index creation time, or all of the above.


Voyager indices support insertion, lookup, and deletion of vectors.

Calling the :py:class:`Index` constructor will return an object of one
of three classes:

- :py:class:`FloatIndex`, which uses 32-bit precision for all data
- :py:class:`Float8Index`, which uses 8-bit fixed-point precision and requires all vectors to be within the bounds [-1, 1]
- :py:class:`E4M3Index`, which uses 8-bit floating-point precision and requires all vectors to be within the bounds [-448, 448]

Args:
    space:
        The :py:class:`Space` to use to calculate the distance between vectors,
        :py:class:`Space.Cosine` (cosine distance).

        The choice of distance to use usually depends on the kind of vector
        data being inserted into the index. If your vectors are usually compared
        by measuring the cosine distance between them, use :py:class:`Space.Cosine`.

    num_dimensions:
        The number of dimensions present in each vector added to this index.
        Each vector in the index must have the same number of dimensions.

    M:
        The number of connections between nodes in the tree's internal data structure.
        Larger values give better recall, but use more memory. This parameter cannot
        be changed after the index is instantiated.

    ef_construction:
        The number of vectors to search through when inserting a new vector into
        the index. Higher values make index construction slower, but give better
        recall. This parameter cannot be changed after the index is instantiated.
    
    random_seed:
        The seed (initial value) of the random number generator used when
        constructing this index. Byte-for-byte identical indices can be created
        by setting this value to a known value and adding vectors to the index
        one-at-a-time (to avoid nondeterministic ordering due to multi-threaded
        insertion).

    max_elements:
        The maximum size of the index at construction time. Indices can be resized
        (and are automatically resized when :py:meth:`add_item` or
        :py:meth:`add_items` is called) so this value is only useful if the exact
        number of elements that will be added to this index is known in advance.
)");

  index.def(
      "add_item",
      [](Index &index,
         std::variant<nb::ndarray<float>, std::vector<float>> vector,
         std::optional<size_t> _id) {
        std::vector<float> stdArray;
        if (std::holds_alternative<nb::ndarray<float>>(vector)) {
          stdArray =
              pyArrayToVector<float>(std::get<nb::ndarray<float>>(vector));
        } else {
          stdArray = std::get<std::vector<float>>(vector);
        }
        nb::gil_scoped_release release;
        return index.addItem(stdArray, _id);
      },
      nb::arg("vector"), nb::arg("id") = nb::none(), R"(
Add a vector to this index.

Args:
    vector: A 32-bit floating-point NumPy array, with shape ``(num_dimensions,)``.

        If using the :py:class:`Space.Cosine` :py:class:`Space`, this vector will be normalized
        before insertion. If using a :py:class:`StorageDataType` other than
        :py:class:`StorageDataType.Float32`, the vector will be converted to the lower-precision
        data type *after* normalization.

    id: An optional ID to assign to this vector.
        If not provided, this vector's ID will automatically be generated based on the
        number of elements already in this index.

Returns:
    The ID that was assigned to this vector (either auto-generated or provided).

.. warning::
    If calling :py:meth:`add_item` in a loop, consider batching your
    calls by using :py:meth:`add_items` instead, which will be faster.
)");

  index.def(
      "add_items",
      [](Index &index, nb::ndarray<float> vectors,
         std::optional<std::vector<size_t>> _ids, int num_threads) {
        std::vector<size_t> empty;
        auto ndArray = pyArrayToNDArray<float, 2>(vectors);

        nb::gil_scoped_release release;
        return index.addItems(ndArray, (_ids ? *_ids : empty), num_threads);
      },
      nb::arg("vectors"), nb::arg("ids") = nb::none(),
      nb::arg("num_threads") = -1,
      R"(
Add multiple vectors to this index simultaneously.

This method may be faster than calling :py:meth:`add_items` multiple times,
as passing a batch of vectors helps avoid Python's Global Interpreter Lock.

Args:
    vectors: A 32-bit floating-point NumPy array, with shape ``(num_vectors, num_dimensions)``.

        If using the :py:class:`Space.Cosine` :py:class:`Space`, these vectors will be normalized
        before insertion. If using a :py:class:`StorageDataType` other than
        :py:class:`StorageDataType.Float32`, these vectors will be converted to the lower-precision
        data type *after* normalization.

    id: An optional list of IDs to assign to these vectors.
        If provided, this list must be identical in length to the first dimension of ``vectors``.
        If not provided, each vector's ID will automatically be generated based on the
        number of elements already in this index.

    num_threads: Up to ``num_threads`` will be started to perform insertions in parallel.
                 If ``vectors`` contains only one query vector, ``num_threads`` will have no effect.
                 By default, one thread will be spawned per CPU core.
Returns:
    The IDs that were assigned to the provided vectors (either auto-generated or provided), in the
    same order as the provided vectors.
)");

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Querying
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  index.def(
      "query",
      [](Index &index, nb::ndarray<float> &input, size_t k = 1,
         int num_threads = -1, long queryEf = -1) {
        int inputNDim = input.ndim();
        switch (inputNDim) {
        case 1: {
          auto idsAndDistances =
              index.query(pyArrayToVector<float>(input), k, queryEf);
          std::tuple<nb::ndarray<hnswlib::labeltype, nb::numpy>,
                     nb::ndarray<float, nb::numpy>>
              output = {vectorToPyArray<hnswlib::labeltype>(
                            std::get<0>(idsAndDistances)),
                        vectorToPyArray<float>(std::get<1>(idsAndDistances))};
          return output;
        }
        case 2: {
          auto idsAndDistances = index.query(pyArrayToNDArray<float, 2>(input),
                                             k, num_threads, queryEf);
          std::tuple<nb::ndarray<hnswlib::labeltype, nb::numpy>,
                     nb::ndarray<float, nb::numpy>>
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
      nb::arg("vectors"), nb::arg("k") = 1, nb::arg("num_threads") = -1,
      nb::arg("query_ef") = -1, R"(
Query this index to retrieve the ``k`` nearest neighbors of the provided vectors.

Args:
    vectors: A 32-bit floating-point NumPy array, with shape ``(num_dimensions,)``
             *or* ``(num_queries, num_dimensions)``.
    
    k: The number of neighbors to return.

    num_threads: If ``vectors`` contains more than one query vector, up
                 to ``num_threads`` will be started to perform queries
                 in parallel. If ``vectors`` contains only one query vector,
                 ``num_threads`` will have no effect. Defaults to using one
                 thread per CPU core.

    query_ef: The depth of search to perform for this query. Up to ``query_ef``
              candidates will be searched through to try to find up the ``k``
              nearest neighbors per query vector.

Returns:
    A tuple of ``(neighbor_ids, distances)``. If a single query vector was provided,
    both ``neighbor_ids`` and ``distances`` will be of shape ``(k,)``.

    If multiple query vectors were provided, both ``neighbor_ids`` and ``distances``
    will be of shape ``(num_queries, k)``, ordered such that the ``i``-th result
    corresponds with the ``i``-th query vector.


Examples
--------

Query with a single vector::

    neighbor_ids, distances = index.query(np.array([1, 2, 3, 4, 5]), k=10)
    neighbor_ids.shape # => (10,)
    distances.shape # => (10,)
  
    for i, (neighbor_id, distance) in enumerate(zip(neighbor_ids, distances)):
        print(f"{i}-th closest neighbor is {neighbor_id}, {distance} away")

Query with multiple vectors simultaneously::

    query_vectors = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ])
    all_neighbor_ids, all_distances = index.query(query_vectors, k=10)
    all_neighbor_ids.shape # => (2, 10)
    all_distances.shape # => (2, 10)
  
    for query_vector, query_neighbor_ids, query_distances in \\
            zip(query_vectors, all_neighbor_ids, all_distances):
        print(f"For query vector {query_vector}:")
        for i, (neighbor_ids, distances) in enumerate(query_neighbor_ids, query_distances):
            print(f"\t{i}-th closest neighbor is {neighbor_id}, {distance} away")

.. warning::

    If using E4M3 storage with the Cosine :py:class:`Space`, some queries may return
    negative distances due to the reduced floating-point precision of the storage
    data type. While confusing, these negative distances still result in a correct
    ordering between results.

)");

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Property Methods
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  index.def_prop_ro("space", &Index::getSpace,
                    "Return the :py:class:`Space` used to calculate "
                    "distances between vectors.");

  index.def_prop_ro("num_dimensions", &Index::getNumDimensions, R"(
The number of dimensions in each vector stored by this index.
)");

  index.def_prop_ro("M", &Index::getM, R"(
The number of connections between nodes in the tree's internal data structure.

Larger values give better recall, but use more memory. This parameter cannot be changed
after the index is instantiated.)");

  index.def_prop_ro("ef_construction", &Index::getEfConstruction, R"(
The number of vectors that this index searches through when inserting a new vector into
the index. Higher values make index construction slower, but give better recall. This
parameter cannot be changed after the index is instantiated.)");

  index.def_prop_rw("max_elements", &Index::getMaxElements, &Index::resizeIndex,
                    R"(
The maximum number of elements that can be stored in this index.

If :py:attr:`max_elements` is much larger than
:py:attr:`num_elements`, this index may use more memory
than necessary. (Call :py:meth:`resize` to reduce the
memory footprint of this index.)

This is a **soft limit**; if a call to :py:meth:`add_item` or
:py:meth:`add_items` would cause :py:attr:`num_elements` to exceed
:py:attr:`max_elements`, the index will automatically be resized
to accommodate the new elements.

Note that assigning to this property is functionally identical to
calling :py:meth:`resize`.
)");

  index.def_prop_ro("storage_data_type", &Index::getStorageDataType,
                    R"(
The :py:class:`StorageDataType` used to store vectors in this :py:class:`Index`.
)");

  index.def_prop_ro("num_elements", &Index::getNumElements, R"(
The number of elements (vectors) currently stored in this index.

Note that the number of elements will not decrease if any elements are
deleted from the index; those deleted elements simply become invisible.)");

  index.def(
      "get_vector",
      [](Index &index, size_t _id) -> nb::ndarray<float, nb::numpy> {
        return ndArrayToPyArray<float, 1>(NDArray<float, 1>(
            index.getVector(_id), {(int)index.getNumDimensions()}));
      },
      nb::arg("id"), R"(
Get the vector stored in this index at the provided integer ID.
If no such vector exists, a :py:exc:`KeyError` will be thrown.

.. note::
    This method can also be called by using the subscript operator
    (i.e. ``my_index[1234]``).

.. warning::
    If using the :py:class:`Cosine` :py:class:`Space`, this method
    will return a normalized version of the vector that was
    originally added to this index.
)");

  index.attr("__getitem__") = index.attr("get_vector");

  index.def(
      "get_vectors",
      [](Index &index, std::vector<size_t> _ids) {
        return ndArrayToPyArray<float, 2>(index.getVectors(_ids));
      },
      nb::arg("ids"), R"(
Get one or more vectors stored in this index at the provided integer IDs.
If one or more of the provided IDs cannot be found in the index, a
:py:exc:`KeyError` will be thrown.

.. warning::
    If using the :py:class:`Cosine` :py:class:`Space`, this method
    will return normalized versions of the vector that were
    originally added to this index.
)");

  index.def_prop_ro(
      "ids",
      [](Index &index) {
        return std::make_unique<LabelSetView>(index.getIDsMap());
      },
      R"(
A set-like object containing the integer IDs stored as 'keys' in this index.

Use these indices to iterate over the vectors in this index, or to check for inclusion of a
specific integer ID in this index::

    index: voyager.Index = ...

    1234 in index.ids  # => true, this ID is present in the index
    1234 in index  # => also works!

    len(index.ids) # => returns the number of non-deleted vectors
    len(index) # => also returns the number of valid labels

    for _id in index.ids:
        print(_id) # print all labels
)");

  index.def(
      "get_distance",
      [](Index &index, std::vector<float> a, std::vector<float> b) {
        return index.getDistance(a, b);
      },
      R"(
Get the distance between two provided vectors. The vectors must share the dimensionality of the index.
)",
      nb::arg("a"), nb::arg("b"));

  index.def(
      "get_distance",
      [](Index &index, nb::ndarray<float> a, nb::ndarray<float> b) {
        return index.getDistance(pyArrayToVector<float>(a),
                                 pyArrayToVector<float>(b));
      },
      R"(
Get the distance between two provided vectors. The vectors must share the dimensionality of the index.
)",
      nb::arg("a"), nb::arg("b"));

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Index Modifier Methods/Attributes
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  index.def_prop_rw("ef", &Index::getEF, &Index::setEF, R"(
The default number of vectors to search through when calling :py:meth:`query`.

Higher values make queries slower, but give better recall.

.. warning::
  Changing this parameter affects all calls to :py:meth:`query` made after this
  change is made.

  This parameter can be overridden on a per-query basis when calling :py:meth:`query`
  by passing the ``query_ef`` parameter, allowing finer-grained control over query
  speed and recall.

)");

  index.def("mark_deleted", &Index::markDeleted, nb::arg("id"), R"(
Mark an ID in this index as deleted.

Deleted IDs will not show up in the results of calls to :py:meth:`query`,
but will still take up space in the index, and will slow down queries.

.. note::
    To delete one or more vectors from a :py:class:`Index` without
    incurring any query-time performance penalty, recreate the index
    from scratch without the vectors you want to remove::
        
        original: Index = ...
        ids_to_remove: Set[int] = ...

        recreated = Index(
          original.space,
          original.num_dimensions,
          original.M,
          original.ef_construction,
          max_elements=len(original),
          storage_data_type=original.storage_data_type
        )
        ordered_ids = list(set(original.ids) - ids_to_remove)
        recreated.add_items(original.get_vectors(ordered_ids), ordered_ids)

.. note::
    This method can also be called by using the ``del`` operator::

        index: voyager.Index = ...
        del index[1234]  # deletes the ID 1234
)");

  index.attr("__delitem__") = index.attr("mark_deleted");

  index.def("unmark_deleted", &Index::unmarkDeleted, nb::arg("id"), R"(
Unmark an ID in this index as deleted.

Once unmarked as deleted, an existing ID will show up in the results of
calls to :py:meth:`query` again.
)");

  index.def("resize", &Index::resizeIndex, nb::arg("new_size"), R"(
Resize this index, allocating space for up to ``new_size`` elements to
be stored. This changes the :py:attr:`max_elements` property and may
cause this :py:class:`Index` object to use more memory. This is a fairly
expensive operation and prevents queries from taking place while the
resize is in process.

Note that the size of an index is a **soft limit**; if a call to
:py:meth:`add_item` or :py:meth:`add_items` would cause
:py:attr:`num_elements` to exceed :py:attr:`max_elements`, the
index will automatically be resized to accommodate the new elements.

Calling :py:meth:`resize` *once* before adding vectors to the
index will speed up index creation if the number of elements is known
in advance, as subsequent calls to :py:meth:`add_items` will not need
to resize the index on-the-fly.
)");

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Save Index
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  static constexpr const char *SAVE_DOCSTRING = R"(
Save this index to the provided file path or file-like object.

If provided a file path, Voyager will release Python's Global Interpreter Lock (GIL)
and will write to the provided file.

If provided a file-like object, Voyager will *not* release the GIL, but will pass
one or more chunks of data (of up to 100MB each) to the provided object for writing.
  )";
  index.def(
      "save",
      [](Index &index, std::string filePath) {
        nb::gil_scoped_release release;
        index.saveIndex(filePath);
      },
      nb::arg("output_path"), SAVE_DOCSTRING);

  index.def(
      "save",
      [](Index &index, nb::object filelike) {
        auto outputStream = std::make_shared<PythonOutputStream>(filelike);

        nb::gil_scoped_release release;
        index.saveIndex(outputStream);
      },
      nb::arg("file_like"), SAVE_DOCSTRING);

  index.def(
      "as_bytes",
      [](Index &index) {
        auto outputStream = std::make_shared<MemoryOutputStream>();
        {
          nb::gil_scoped_release release;
          index.saveIndex(outputStream);
        }

        std::string bytes = outputStream->getValue();
        return nb::bytes(bytes.data(), bytes.size());
      },
      R"(
Returns the contents of this index as a :py:class:`bytes` object. The resulting object
will contain the same data as if this index was serialized to disk and then read back
into memory again.

.. warning::
    This may be extremely large (many gigabytes) if the index is sufficiently large.
    To save to disk without allocating this entire bytestring, use :py:meth:`save`.

.. note::
    This method can also be called by passing this object to the ``bytes(...)``
    built-in function::

        index: voyager.Index = ...
        serialized_index = bytes(index)

)");

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Python Builtin Supports
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  index.attr("__bytes__") = index.attr("as_bytes");

  index.def(
      "__contains__",
      [](Index &self, hnswlib::labeltype element) {
        auto &map = self.getIDsMap();
        return map.find(element) != map.end();
      },
      nb::arg("id"),
      R"(
Check to see if a provided vector's ID is present in this index.

Returns true iff the provided integer ID has a corresponding (non-deleted) vector in this index.
Use the ``in`` operator to call this method::

    1234 in index # => returns True or False
)");

  index.def(
      "__len__", [](Index &self) { return self.getIDsMap().size(); }, R"(
Returns the number of non-deleted vectors in this index.

Use the ``len`` operator to call this method::

    len(index) # => 1234

.. note::
    This value may differ from :py:attr:`num_elements` if elements have been deleted.
)");

  register_index_class<float, float>(
      m, "FloatIndex",
      "An :py:class:`Index` that uses full-precision 32-bit floating-point "
      "storage.");
  // An int8 index that stores its values as 8-bit integers, assumes all
  // input/output data is float on [-1, 1], and returns floating-point
  // distances.
  register_index_class<float, int8_t, std::ratio<1, 127>>(
      m, "Float8Index",
      "An :py:class:`Index` that uses fixed-point 8-bit storage.");

  // An 8-bit floating-point index class that has even more reduced
  // precision over Float8, but allows values on the range [-448, 448].
  // Inspired by: https://arxiv.org/pdf/2209.05433.pdf
  register_index_class<float, E4M3>(
      m, "E4M3Index",
      "An :py:class:`Index` that uses floating-point 8-bit storage.");

  index.def_static(
      "__new__",
      [](const nb::object *, const SpaceType space, const int num_dimensions,
         const size_t M, const size_t ef_construction, const size_t random_seed,
         const size_t max_elements,
         const StorageDataType storageDataType) -> std::shared_ptr<Index> {
        nb::gil_scoped_release release;
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
      nb::arg("cls"), nb::arg("space"), nb::arg("num_dimensions"),
      nb::arg("M") = 12, nb::arg("ef_construction") = 200,
      nb::arg("random_seed") = 1, nb::arg("max_elements") = 1,
      nb::arg("storage_data_type") = StorageDataType::Float32,
      R"(
Create a new Voyager nearest-neighbor search index with the provided arguments.

See documentation for :py:meth:`Index.__init__` for details on required arguments.
)");

  static constexpr const char *LOAD_DOCSTRING = R"(
Load an index from a file on disk, or a Python file-like object.

If provided a string as a first argument, the string is assumed to refer to a file
on the local filesystem. Loading of an index from this file will be done in native
code, without holding Python's Global Interpreter Lock (GIL), allowing for performant
loading of multiple indices simultaneously.

If provided a file-like object as a first argument, the provided object must have
``read``, ``seek``, ``tell``, and ``seekable`` methods, and must return
binary data (i.e.: ``open(..., \"rb\")`` or ``io.BinaryIO``, etc.).

The additional arguments for :py:class:`Space`, ``num_dimensions``, and
:py:class:`StorageDataType` allow for loading of index files created with versions
of Voyager prior to v1.3.

.. warning::
    Loading an index from a file-like object will not release the GIL.
    However, chunks of data of up to 100MB in size will be read from the file-like
    object at once, hopefully reducing the impact of the GIL.
)";

  index.def_static(
      "load",
      [](const std::string filename, const SpaceType space,
         const int num_dimensions,
         const StorageDataType storageDataType) -> std::shared_ptr<Index> {
        nb::gil_scoped_release release;

        auto inputStream = std::make_shared<FileInputStream>(filename);
        std::unique_ptr<voyager::Metadata::V1> metadata =
            voyager::Metadata::loadFromStream(inputStream);

        if (metadata) {
          if (metadata->getStorageDataType() != storageDataType) {
            throw std::domain_error(
                "Provided storage data type (" + toString(storageDataType) +
                ") does not match the data type used in this file (" +
                toString(metadata->getStorageDataType()) + ").");
          }
          if (metadata->getSpaceType() != space) {
            throw std::domain_error(
                "Provided space type (" + toString(space) +
                ") does not match the space type used in this file (" +
                toString(metadata->getSpaceType()) + ").");
          }
          if (metadata->getNumDimensions() != num_dimensions) {
            throw std::domain_error(
                "Provided number of dimensions (" +
                std::to_string(num_dimensions) +
                ") does not match the number of dimensions used in this file "
                "(" +
                std::to_string(metadata->getNumDimensions()) + ").");
          }

          return loadTypedIndexFromMetadata(std::move(metadata), inputStream);
        }

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
      nb::arg("filename"), nb::arg("space"), nb::arg("num_dimensions"),
      nb::arg("storage_data_type") = StorageDataType::Float32, LOAD_DOCSTRING);

  index.def_static(
      "load",
      [](const std::string filename) -> std::shared_ptr<Index> {
        nb::gil_scoped_release release;

        return loadTypedIndexFromStream(
            std::make_shared<FileInputStream>(filename));
      },
      nb::arg("filename"), LOAD_DOCSTRING);

  index.def_static(
      "load",
      [](const nb::object filelike, const SpaceType space,
         const int num_dimensions,
         const StorageDataType storageDataType) -> std::shared_ptr<Index> {
        if (!isReadableFileLike(filelike)) {
          throw nb::type_error(
              ("Expected either a filename or a file-like object (with "
               "read, seek, seekable, and tell methods), but received: " +
               nb::cast<std::string>(filelike.attr("__repr__")()))
                  .c_str());
        }

        auto inputStream = std::make_shared<PythonInputStream>(filelike);
        nb::gil_scoped_release release;

        std::unique_ptr<voyager::Metadata::V1> metadata =
            voyager::Metadata::loadFromStream(inputStream);

        if (metadata) {
          if (metadata->getStorageDataType() != storageDataType) {
            throw std::domain_error(
                "Provided storage data type (" + toString(storageDataType) +
                ") does not match the data type used in this file (" +
                toString(metadata->getStorageDataType()) + ").");
          }
          if (metadata->getSpaceType() != space) {
            throw std::domain_error(
                "Provided space type (" + toString(space) +
                ") does not match the space type used in this file (" +
                toString(metadata->getSpaceType()) + ").");
          }
          if (metadata->getNumDimensions() != num_dimensions) {
            throw std::domain_error(
                "Provided number of dimensions (" +
                std::to_string(num_dimensions) +
                ") does not match the number of dimensions used in this file "
                "(" +
                std::to_string(metadata->getNumDimensions()) + ").");
          }
          return loadTypedIndexFromMetadata(std::move(metadata), inputStream);
        }

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
      nb::arg("file_like"), nb::arg("space"), nb::arg("num_dimensions"),
      nb::arg("storage_data_type") = StorageDataType::Float32, LOAD_DOCSTRING);

  index.def_static(
      "load",
      [](const nb::object filelike) -> std::shared_ptr<Index> {
        if (!isReadableFileLike(filelike)) {
          throw nb::type_error(
              ("Expected either a filename or a file-like object (with "
               "read, seek, seekable, and tell methods), but received: " +
               nb::cast<std::string>(filelike.attr("__repr__")()))
                  .c_str());
        }

        auto inputStream = std::make_shared<PythonInputStream>(filelike);
        nb::gil_scoped_release release;

        return loadTypedIndexFromStream(inputStream);
      },
      nb::arg("file_like"), LOAD_DOCSTRING);
}
