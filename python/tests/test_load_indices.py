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
import numpy as np
from glob import glob

from voyager import Index, Space, StorageDataType

INDEX_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "indices")


@pytest.mark.parametrize("load_from_stream", [False, True])
@pytest.mark.parametrize("index_filename", glob(os.path.join(INDEX_FIXTURE_DIR, "v0", "*.hnsw")))
def test_load_v0_indices(load_from_stream: bool, index_filename: str):
    space = Space.Euclidean
    if "cosine" in index_filename:
        space = Space.Cosine
    elif "innerproduct" in index_filename:
        space = Space.InnerProduct
    elif "euclidean" in index_filename:
        space = Space.Euclidean
    else:
        raise ValueError(f"Not sure which space type is used in {index_filename}")

    num_dimensions = int(index_filename.split("_")[1].split("dim")[0])
    storage_data_type = index_filename.split("_")[-1].split(".")[0].lower()
    if storage_data_type == "float32":
        storage_data_type = StorageDataType.Float32
    elif storage_data_type == "float8":
        storage_data_type = StorageDataType.Float8
    elif storage_data_type == "e4m3":
        storage_data_type = StorageDataType.E4M3
    else:
        raise ValueError(f"Not sure which storage data type is used in {index_filename}")

    if load_from_stream:
        with open(index_filename, "rb") as f:
            index = Index.load(
                f,
                space=space,
                num_dimensions=num_dimensions,
                storage_data_type=storage_data_type,
            )
    else:
        index = Index.load(
            index_filename,
            space=space,
            num_dimensions=num_dimensions,
            storage_data_type=storage_data_type,
        )

    # All of these test indices are expected to contain exactly 0.0, 0.1, 0.2, 0.3, 0.4
    assert set(index.ids) == {0, 1, 2, 3, 4}
    for _id in index.ids:
        expected_vector = np.ones(num_dimensions) * (_id * 0.1)
        if space == Space.Cosine and _id > 0:
            # Voyager stores only normalized vectors in Cosine mode:
            expected_vector = expected_vector / np.sqrt(np.sum(expected_vector**2))
        np.testing.assert_allclose(index[_id], expected_vector, atol=0.2)
