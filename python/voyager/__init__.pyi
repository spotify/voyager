from __future__ import annotations
import voyager

import typing

original_overload = typing.overload
__OVERLOADED_DOCSTRINGS = {}

def patch_overload(func):
    original_overload(func)
    if func.__doc__:
        __OVERLOADED_DOCSTRINGS[func.__qualname__] = func.__doc__
    else:
        func.__doc__ = __OVERLOADED_DOCSTRINGS.get(func.__qualname__)
    if func.__doc__:
        # Work around the fact that pybind11-stubgen generates
        # duplicate docstrings sometimes, once for each overload:
        while func.__doc__[len(func.__doc__) // 2 :].strip() == func.__doc__[: -len(func.__doc__) // 2].strip():
            func.__doc__ = func.__doc__[len(func.__doc__) // 2 :].strip()
    return func

typing.overload = patch_overload

from typing_extensions import Literal
from enum import Enum
import numpy

_Shape = typing.Tuple[int, ...]

__all__ = [
    "Cosine",
    "E4M3",
    "E4M3Index",
    "E4M3T",
    "Euclidean",
    "Float32",
    "Float8",
    "Float8Index",
    "FloatIndex",
    "Index",
    "InnerProduct",
    "LabelSetView",
    "Space",
    "StorageDataType",
]

class Space(Enum):
    """
    The method used to calculate the distance between vectors.
    """

    Euclidean = 0  # fmt: skip
    """
    Euclidean distance; also known as L2 distance. The square root of the sum of the squared differences between each element of each vector.
    """
    InnerProduct = 1  # fmt: skip
    """
    Inner product distance.
    """
    Cosine = 2  # fmt: skip
    """
    Cosine distance; also known as normalized inner product.
    """

class StorageDataType(Enum):
    """
    The data type used to store vectors in memory and on-disk.

    The :py:class:`StorageDataType` used for an :py:class:`Index` directly determines
    its memory usage, disk space usage, and recall. Both :py:class:`Float8` and
    :py:class:`E4M3` data types use 8 bits (1 byte) per dimension per vector, reducing
    memory usage and index size by a factor of 4 compared to :py:class:`Float32`.
    """

    Float8 = 16  # fmt: skip
    """
    8-bit fixed-point decimal values. All values must be within [-1, 1.00787402].
    """
    Float32 = 32  # fmt: skip
    """
    32-bit floating point (default).
    """
    E4M3 = 48  # fmt: skip
    """
    8-bit floating point with a range of [-448, 448], from the paper "FP8 Formats for Deep Learning" by Micikevicius et al.

    .. warning::
        Using E4M3 with the Cosine :py:class:`Space` may cause some queries to return negative distances due to the reduced floating-point precision. While confusing, these negative distances still result in a correct ordering between results.
    """

Cosine: voyager.Space  # value = <Space.Cosine: 2>
E4M3: voyager.StorageDataType  # value = <StorageDataType.E4M3: 48>
Euclidean: voyager.Space  # value = <Space.Euclidean: 0>
Float32: voyager.StorageDataType  # value = <StorageDataType.Float32: 32>
Float8: voyager.StorageDataType  # value = <StorageDataType.Float8: 16>
InnerProduct: voyager.Space  # value = <Space.InnerProduct: 1>

class Index:
    """
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
    """

    def __contains__(self, id: int) -> bool:
        """
        Check to see if a provided vector's ID is present in this index.

        Returns true iff the provided integer ID has a corresponding (non-deleted) vector in this index.
        Use the ``in`` operator to call this method::

            1234 in index # => returns True or False
        """

    def __len__(self) -> int:
        """
        Returns the number of non-deleted vectors in this index.

        Use the ``len`` operator to call this method::

            len(index) # => 1234

        .. note::
            This value may differ from :py:attr:`num_elements` if elements have been deleted.
        """

    @classmethod
    def __new__(
        cls,
        space: Space,
        num_dimensions: int,
        M: int = 12,
        ef_construction: int = 200,
        random_seed: int = 1,
        max_elements: int = 1,
        storage_data_type: StorageDataType = StorageDataType.Float32,
    ) -> Index:
        """
        Create a new Voyager nearest-neighbor search index with the provided arguments.

        See documentation for :py:meth:`Index.__init__` for details on required arguments.
        """

    def add_item(
        self,
        vector: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]],
        id: typing.Optional[int] = None,
    ) -> int:
        """
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
        """

    def add_items(
        self,
        vectors: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]],
        ids: typing.Optional[typing.List[int]] = None,
        num_threads: int = -1,
    ) -> typing.List[int]:
        """
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
        """

    def as_bytes(self) -> bytes:
        """
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
        """

    def get_distance(self, a: typing.List[float], b: typing.List[float]) -> float:
        """
        Get the distance between two provided vectors. The vectors must share the dimensionality of the index.
        """

    def get_vector(self, id: int) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        Get the vector stored in this index at the provided integer ID.
        If no such vector exists, a :py:exc:`KeyError` will be thrown.

        .. note::
            This method can also be called by using the subscript operator
            (i.e. ``my_index[1234]``).

        .. warning::
            If using the :py:class:`Cosine` :py:class:`Space`, this method
            will return a normalized version of the vector that was
            originally added to this index.
        """

    def get_vectors(self, ids: typing.List[int]) -> numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]]:
        """
        Get one or more vectors stored in this index at the provided integer IDs.
        If one or more of the provided IDs cannot be found in the index, a
        :py:exc:`KeyError` will be thrown.

        .. warning::
            If using the :py:class:`Cosine` :py:class:`Space`, this method
            will return normalized versions of the vector that were
            originally added to this index.
        """

    @staticmethod
    @typing.overload
    def load(
        filename: str,
        space: Space,
        num_dimensions: int,
        storage_data_type: StorageDataType = StorageDataType.Float32,
    ) -> Index:
        """
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
        """

    @staticmethod
    @typing.overload
    def load(filename: str) -> Index: ...
    @staticmethod
    @typing.overload
    def load(
        file_like: typing.BinaryIO,
        space: Space,
        num_dimensions: int,
        storage_data_type: StorageDataType = StorageDataType.Float32,
    ) -> Index: ...
    @staticmethod
    @typing.overload
    def load(file_like: typing.BinaryIO) -> Index: ...
    def mark_deleted(self, id: int) -> None:
        """
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
        """

    def query(
        self,
        vectors: numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]],
        k: int = 1,
        num_threads: int = -1,
        query_ef: int = -1,
    ) -> typing.Tuple[
        numpy.ndarray[typing.Any, numpy.dtype[numpy.uint64]],
        numpy.ndarray[typing.Any, numpy.dtype[numpy.float32]],
    ]:
        """
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
        """

    def resize(self, new_size: int) -> None:
        """
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
        """

    @typing.overload
    def save(self, output_path: str) -> None:
        """
        Save this index to the provided file path or file-like object.

        If provided a file path, Voyager will release Python's Global Interpreter Lock (GIL)
        and will write to the provided file.

        If provided a file-like object, Voyager will *not* release the GIL, but will pass
        one or more chunks of data (of up to 100MB each) to the provided object for writing.



        Save this index to the provided file path or file-like object.

        If provided a file path, Voyager will release Python's Global Interpreter Lock (GIL)
        and will write to the provided file.

        If provided a file-like object, Voyager will *not* release the GIL, but will pass
        one or more chunks of data (of up to 100MB each) to the provided object for writing.

        """

    @typing.overload
    def save(self, file_like: typing.BinaryIO) -> None: ...
    def unmark_deleted(self, id: int) -> None:
        """
        Unmark an ID in this index as deleted.

        Once unmarked as deleted, an existing ID will show up in the results of
        calls to :py:meth:`query` again.
        """

    @property
    def M(self) -> int:
        """
        The number of connections between nodes in the tree's internal data structure.

        Larger values give better recall, but use more memory. This parameter cannot be changed
        after the index is instantiated.


        """

    @property
    def ef(self) -> int:
        """
        The default number of vectors to search through when calling :py:meth:`query`.

        Higher values make queries slower, but give better recall.

        .. warning::
          Changing this parameter affects all calls to :py:meth:`query` made after this
          change is made.

          This parameter can be overridden on a per-query basis when calling :py:meth:`query`
          by passing the ``query_ef`` parameter, allowing finer-grained control over query
          speed and recall.


        """

    @ef.setter
    def ef(self, arg1: int) -> None:
        """
        The default number of vectors to search through when calling :py:meth:`query`.

        Higher values make queries slower, but give better recall.

        .. warning::
          Changing this parameter affects all calls to :py:meth:`query` made after this
          change is made.

          This parameter can be overridden on a per-query basis when calling :py:meth:`query`
          by passing the ``query_ef`` parameter, allowing finer-grained control over query
          speed and recall.
        """

    @property
    def ef_construction(self) -> int:
        """
        The number of vectors that this index searches through when inserting a new vector into
        the index. Higher values make index construction slower, but give better recall. This
        parameter cannot be changed after the index is instantiated.


        """

    @property
    def ids(self) -> LabelSetView:
        """
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


        """

    @property
    def max_elements(self) -> int:
        """
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


        """

    @max_elements.setter
    def max_elements(self, arg1: int) -> None:
        """
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
        """

    @property
    def num_dimensions(self) -> int:
        """
        The number of dimensions in each vector stored by this index.


        """

    @property
    def num_elements(self) -> int:
        """
        The number of elements (vectors) currently stored in this index.

        Note that the number of elements will not decrease if any elements are
        deleted from the index; those deleted elements simply become invisible.


        """

    @property
    def space(self) -> Space:
        """
        Return the :py:class:`Space` used to calculate distances between vectors.


        """

    @property
    def storage_data_type(self) -> StorageDataType:
        """
        The :py:class:`StorageDataType` used to store vectors in this :py:class:`Index`.


        """
    pass

class E4M3T:
    """
    An 8-bit floating point data type with reduced precision and range. This class wraps a C++ struct and should probably not be used directly.
    """

    def __float__(self) -> float:
        """
        Cast the given E4M3 number to a float.
        """

    @typing.overload
    def __init__(self, value: float) -> None:
        """
        Create an E4M3 number given a floating-point value. If out of range, the value will be clipped.

        Create an E4M3 number given a sign, exponent, and mantissa. If out of range, the values will be clipped.
        """

    @typing.overload
    def __init__(self, sign: int, exponent: int, mantissa: int) -> None: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def from_char(value: int) -> E4M3T:
        """
        Create an E4M3 number given a raw 8-bit value.
        """

    @property
    def exponent(self) -> int:
        """
        The effective exponent of this E4M3 number, expressed as an integer.


        """

    @property
    def mantissa(self) -> float:
        """
        The effective mantissa (non-exponent part) of this E4M3 number, expressed as an integer.


        """

    @property
    def raw_exponent(self) -> int:
        """
        The raw value of the exponent part of this E4M3 number, expressed as an integer.


        """

    @property
    def raw_mantissa(self) -> int:
        """
        The raw value of the mantissa (non-exponent part) of this E4M3 number, expressed as a floating point value.


        """

    @property
    def sign(self) -> int:
        """
        The sign bit from this E4M3 number. Will be ``1`` if the number is negative, ``0`` otherwise.


        """

    @property
    def size(self) -> int:
        """
        The number of bytes used to represent this (C++) instance in memory.


        """
    pass

class Float8Index(Index):
    """
    An :py:class:`Index` that uses fixed-point 8-bit storage.
    """

    def __init__(
        self,
        space: Space,
        num_dimensions: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 1,
        max_elements: int = 1,
    ) -> None:
        """
        Create a new, empty index.
        """

    def __repr__(self) -> str: ...
    pass

class FloatIndex(Index):
    """
    An :py:class:`Index` that uses full-precision 32-bit floating-point storage.
    """

    def __init__(
        self,
        space: Space,
        num_dimensions: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 1,
        max_elements: int = 1,
    ) -> None:
        """
        Create a new, empty index.
        """

    def __repr__(self) -> str: ...
    pass

class E4M3Index(Index):
    """
    An :py:class:`Index` that uses floating-point 8-bit storage.
    """

    def __init__(
        self,
        space: Space,
        num_dimensions: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 1,
        max_elements: int = 1,
    ) -> None:
        """
        Create a new, empty index.
        """

    def __repr__(self) -> str: ...
    pass

class LabelSetView:
    """
    A read-only set-like object containing 64-bit integers. Use this object like a regular Python :py:class:`set` object, by either iterating through it, or checking for membership with the ``in`` operator.
    """

    @typing.overload
    def __contains__(self, id: int) -> bool: ...
    @typing.overload
    def __contains__(self, id: object) -> bool: ...
    def __iter__(self) -> object: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    pass
