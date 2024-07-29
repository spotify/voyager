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

import numpy as np

import voyager


@pytest.mark.parametrize(
    "num_dimensions,num_elements",
    [
        (4, 1_024),
        (16, 1_024),
        (128, 512),
        (256, 256),
        (4096, 128),
    ],
)
@pytest.mark.parametrize(
    "space",
    [voyager.Space.Euclidean, voyager.Space.Cosine],
    ids=lambda x: x.name if hasattr(x, "name") else str(x),
)
@pytest.mark.parametrize(
    "storage_data_type",
    [
        voyager.StorageDataType.E4M3,
        voyager.StorageDataType.Float8,
        voyager.StorageDataType.Float32,
    ],
    ids=lambda x: x.name if hasattr(x, "name") else str(x),
)
def test_recreate_index(
    num_dimensions: int,
    num_elements: int,
    space: voyager.Space,
    storage_data_type: voyager.StorageDataType,
):
    input_data = np.random.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1

    index = voyager.Index(
        space=space,
        num_dimensions=num_dimensions,
        ef_construction=num_elements,
        M=20,
        storage_data_type=storage_data_type,
    )

    ids = index.add_items(input_data)
    assert sorted(ids) == sorted(index.ids)

    recreated = voyager.Index(
        index.space,
        index.num_dimensions,
        index.M,
        index.ef_construction,
        max_elements=len(index),
        storage_data_type=index.storage_data_type,
    )
    ordered_ids = list(index.ids)
    recreated.add_items(index.get_vectors(ordered_ids), ordered_ids)

    for _id in ids:
        assert _id in index
        assert _id in recreated
        np.testing.assert_allclose(
            index[_id],
            recreated[_id],
            0.08,
            # E4M3 normalization is not idempotent:
            0.1 if storage_data_type == voyager.StorageDataType.E4M3 else 1e-2,
        )
