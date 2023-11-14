#! /usr/bin/env python
#
# Copyright 2022-2023 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import os
import struct
from io import BytesIO
import numpy as np
from glob import glob

from voyager import Index, Space, StorageDataType

INDEX_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "indices")


def detect_space_from_filename(filename: str):
    if "cosine" in filename:
        return Space.Cosine
    elif "innerproduct" in filename:
        return Space.InnerProduct
    elif "euclidean" in filename:
        return Space.Euclidean
    else:
        raise ValueError(f"Not sure which space type is used in {filename}")


def detect_num_dimensions_from_filename(filename: str) -> int:
    return int(filename.split("_")[1].split("dim")[0])


def detect_storage_datatype_from_filename(filename: str) -> int:
    storage_data_type = filename.split("_")[-1].split(".")[0].lower()
    if storage_data_type == "float32":
        return StorageDataType.Float32
    elif storage_data_type == "float8":
        return StorageDataType.Float8
    elif storage_data_type == "e4m3":
        return StorageDataType.E4M3
    else:
        raise ValueError(f"Not sure which storage data type is used in {filename}")


@pytest.mark.parametrize("load_from_stream", [False, True])
@pytest.mark.parametrize(
    "index_filename",
    # Both V0 and V1 indices should be loadable with this interface:
    list(glob(os.path.join(INDEX_FIXTURE_DIR, "v0", "*.hnsw"))) + glob(os.path.join(INDEX_FIXTURE_DIR, "v1", "*.hnsw")),
)
def test_load_v0_indices(load_from_stream: bool, index_filename: str):
    space = detect_space_from_filename(index_filename)
    num_dimensions = detect_num_dimensions_from_filename(index_filename)
    if load_from_stream:
        with open(index_filename, "rb") as f:
            index = Index.load(
                f,
                space=space,
                num_dimensions=num_dimensions,
                storage_data_type=detect_storage_datatype_from_filename(index_filename),
            )
    else:
        index = Index.load(
            index_filename,
            space=space,
            num_dimensions=num_dimensions,
            storage_data_type=detect_storage_datatype_from_filename(index_filename),
        )

    # All of these test indices are expected to contain exactly 0.0, 0.1, 0.2, 0.3, 0.4
    assert set(index.ids) == {0, 1, 2, 3, 4}
    for _id in index.ids:
        expected_vector = np.ones(num_dimensions) * (_id * 0.1)
        if space == Space.Cosine and _id > 0:
            # Voyager stores only normalized vectors in Cosine mode:
            expected_vector = expected_vector / np.sqrt(np.sum(expected_vector**2))
        np.testing.assert_allclose(index[_id], expected_vector, atol=0.2)


@pytest.mark.parametrize("load_from_stream", [False, True])
@pytest.mark.parametrize("index_filename", glob(os.path.join(INDEX_FIXTURE_DIR, "v1", "*.hnsw")))
def test_load_v1_indices(load_from_stream: bool, index_filename: str):
    space = detect_space_from_filename(index_filename)
    num_dimensions = detect_num_dimensions_from_filename(index_filename)
    if load_from_stream:
        with open(index_filename, "rb") as f:
            index = Index.load(f)
    else:
        index = Index.load(index_filename)

    # All of these test indices are expected to contain exactly 0.0, 0.1, 0.2, 0.3, 0.4
    assert set(index.ids) == {0, 1, 2, 3, 4}
    for _id in index.ids:
        expected_vector = np.ones(num_dimensions) * (_id * 0.1)
        if space == Space.Cosine and _id > 0:
            # Voyager stores only normalized vectors in Cosine mode:
            expected_vector = expected_vector / np.sqrt(np.sum(expected_vector**2))
        np.testing.assert_allclose(index[_id], expected_vector, atol=0.2)


@pytest.mark.parametrize("load_from_stream", [False, True])
@pytest.mark.parametrize("index_filename", glob(os.path.join(INDEX_FIXTURE_DIR, "v1", "*.hnsw")))
def test_v1_indices_must_have_no_parameters_or_must_match(load_from_stream: bool, index_filename: str):
    space = detect_space_from_filename(index_filename)
    num_dimensions = detect_num_dimensions_from_filename(index_filename)
    storage_data_type = detect_storage_datatype_from_filename(index_filename)
    with pytest.raises(ValueError) as exception:
        if load_from_stream:
            with open(index_filename, "rb") as f:
                Index.load(
                    f,
                    space=space,
                    num_dimensions=num_dimensions + 1,
                    storage_data_type=storage_data_type,
                )
        else:
            Index.load(
                index_filename,
                space=space,
                num_dimensions=num_dimensions + 1,
                storage_data_type=storage_data_type,
            )
    assert "number of dimensions" in repr(exception)
    assert f"({num_dimensions})" in repr(exception)
    assert f"({num_dimensions + 1})" in repr(exception)


@pytest.mark.parametrize(
    "data,should_pass",
    [
        (
            b"VOYA"  # Header
            b"\x01\x00\x00\x00"  # File version
            b"\x0A\x00\x00\x00"  # Number of dimensions (10)
            b"\x00"  # Space type
            b"\x20"
            + struct.pack("f", 0)  # Storage data type and maximum norm
            + b"\x00",  # Use order-preserving transform
            False,
        ),
        (
            b"VOYA"  # Header
            b"\x01\x00\x00\x00"  # File version
            b"\x0A\x00\x00\x00"  # Number of dimensions (10)
            b"\x00"  # Space type
            b"\x20"  # Storage data type
            + struct.pack("f", 0)  # maximum norm
            + b"\x00"  # Use order-preserving transform
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # offsetLevel0_
            b"\x01\x00\x00\x00\x00\x00\x00\x00"  # max_elements_
            b"\x01\x00\x00\x00\x00\x00\x00\x00"  # cur_element_count
            b"\x34\x00\x00\x00\x00\x00\x00\x00"  # size_data_per_element_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # label_offset_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # offsetData_
            b"\x00\x00\x00\x00"  # maxlevel_
            b"\x00\x00\x00\x00"  # enterpoint_node_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # maxM_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # maxM0_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # M_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # mult_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # ef_construction_
            + (b"\x00" * 52)  # one vector
            + b"\x00\x00\x00\x00",  # one linklist
            True,
        ),
        (
            b"VOYA"  # Header
            b"\x01\x00\x00\x00"  # File version
            b"\x0A\x00\x00\x00"  # Number of dimensions (10)
            b"\x00"  # Space type
            b"\x20"  # Storage data type
            + struct.pack("f", 0)  # maximum norm
            + b"\x00"  # Use order-preserving transform
            b"\x00\x00\x00\xFF\x00\x00\x00\x00"  # offsetLevel0_
            b"\x01\x00\x00\x00\x00\x00\x00\x00"  # max_elements_
            b"\x01\x00\x00\x00\x00\x00\x00\x00"  # cur_element_count
            b"\x34\x00\x00\x00\x00\x00\x00\x00"  # size_data_per_element_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # label_offset_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # offsetData_
            b"\x00\x00\x00\x00"  # maxlevel_
            b"\x00\x00\x00\x00"  # enterpoint_node_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # maxM_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # maxM0_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # M_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # mult_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # ef_construction_
            + (b"\x00" * 52)  # one vector
            + (b"\x00\x00\x00\x00"),  # one linklist
            False,
        ),
        (
            b"VOYA"  # Header
            b"\x01\x00\x00\x00"  # File version
            b"\x0A\x00\x00\x00"  # Number of dimensions (10)
            b"\x00"  # Space type
            b"\x20"  # Storage data type
            + struct.pack("f", 0)  # maximum norm
            + b"\x00"  # Use order-preserving transform
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # offsetLevel0_
            b"\x02\x00\x00\x00\x00\x00\x00\x00"  # max_elements_
            b"\x02\x00\x00\x00\x00\x00\x00\x00"  # cur_element_count
            b"\x48\x00\x00\x00\x00\x00\x00\x00"  # size_data_per_element_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # label_offset_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # offsetData_
            b"\x05\x00\x00\x00"  # maxlevel_
            b"\x05\x00\x00\x00"  # enterpoint_node_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # maxM_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # maxM0_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # M_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # mult_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # ef_construction_
            + (b"\x01" * 72)  # one vector
            + (b"\x01\x00\x00\x00" * 20)  # one linklist
            + b"\x00",
            False,
        ),
        (
            b"VOYA"  # Header
            b"\x01\x00\x00\x00"  # File version
            b"\x0A\x00\x00\x00"  # Number of dimensions (10)
            b"\x00"  # Space type
            b"\x20"  # Storage data type
            + struct.pack("f", 0)  # maximum norm
            + b"\x00"  # Use order-preserving transform
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # offsetLevel0_
            b"\x02\x00\x00\x00\x00\x00\x00\x00"  # max_elements_
            b"\x02\x00\x00\x00\x00\x00\x00\x00"  # cur_element_count
            b"\x48\x00\x00\x00\x00\x00\x00\x00"  # size_data_per_element_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # label_offset_
            b"\x00\x00\x00\x00\x00\x00\x00\x00"  # offsetData_
            b"\x05\x00\x00\x00"  # maxlevel_
            b"\x01\x00\x00\x00"  # enterpoint_node_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # maxM_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # maxM0_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # M_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # mult_
            b"\x05\x00\x00\x00\x00\x00\x00\x00"  # ef_construction_
            + (b"\x01" * 72)  # one vector
            + (b"\x01\x00\x00\x00" * 20)  # one linklist
            + b"\x00",
            False,
        ),
    ],
)
def test_loading_invalid_data_cannot_crash(data: bytes, should_pass: bool):
    if should_pass:
        index = Index.load(BytesIO(data))
        assert len(index) == 1
        np.testing.assert_allclose(index[0], np.zeros(index.num_dimensions))
    else:
        with pytest.raises(Exception):
            index = Index.load(BytesIO(data))
            # We shoulnd't get here, but if we do: do we segfault?
            for id in index.ids:
                index.query(index[id])


@pytest.mark.parametrize("seed", range(1000))
@pytest.mark.parametrize(
    "with_valid_header,offset_level_0",
    [(True, 500_000), (True, None), (False, None)],
)
def test_fuzz(seed: int, with_valid_header: bool, offset_level_0: int):
    """
    Send in 10,000 randomly-generated indices to ensure that the process doesn't crash
    """
    np.random.seed(seed)
    num_bytes = np.random.randint(1_000_000)
    random_data = BytesIO((np.random.rand(num_bytes) * 255).astype(np.uint8).tobytes())
    if with_valid_header:
        random_data.seek(0)
        random_data.write(
            b"VOYA"  # Header
            b"\x01\x00\x00\x00"  # File version
            b"\x0A\x00\x00\x00"  # Number of dimensions (10)
            b"\x00"  # Space type
            b"\x20"  # Storage data type
        )
    if offset_level_0:
        random_data.write(struct.pack("=Q", offset_level_0))
    random_data.seek(0)
    with pytest.raises(Exception):
        Index.load(random_data)
