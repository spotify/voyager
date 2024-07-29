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

import numpy as np
import pytest
from voyager import E4M3T, Index, Space, StorageDataType


def normalized(vec: np.ndarray) -> np.ndarray:
    return vec / (np.sqrt(np.sum(np.power(vec, 2))) + 1e-30)


def inner_product_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - np.dot(a, b)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return inner_product_distance(normalized(a), normalized(b))


def l2_square(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.power(a - b, 2))


def quantize_to_float8(vec: np.ndarray) -> np.ndarray:
    return (vec * 127).astype(np.int8).astype(np.float32) / 127.0


def quantize_to_e4m3(vec: np.ndarray) -> np.ndarray:
    return np.array([float(E4M3T(x)) for x in vec])


@pytest.mark.parametrize("dimensions", [1, 2, 5, 7, 13, 40, 100])
@pytest.mark.parametrize("space", Space.__members__.values(), ids=lambda x: x.name)
@pytest.mark.parametrize(
    "storage_data_type,tolerance",
    [
        (StorageDataType.Float32, 1e-5),
        (StorageDataType.Float8, 0.1),
        (StorageDataType.E4M3, 0.2),
    ],
    ids=str,
)
def test_distance(dimensions: int, space: Space, storage_data_type: StorageDataType, tolerance: float):
    index = Index(space=space, num_dimensions=dimensions, storage_data_type=storage_data_type)
    a = np.random.rand(dimensions)
    b = np.random.rand(dimensions)

    actual = index.get_distance(a, b)

    if space == Space.Cosine:
        a, b = normalized(a), normalized(b)

    if storage_data_type == StorageDataType.Float8:
        a, b = quantize_to_float8(a), quantize_to_float8(b)
    elif storage_data_type == StorageDataType.E4M3:
        a, b = quantize_to_e4m3(a), quantize_to_e4m3(b)

    if space == Space.Cosine:
        # Don't re-normalize here, as we may have already
        # quantized to a lower-precision datatype:
        expected = inner_product_distance(a, b)
    elif space == Space.InnerProduct:
        expected = inner_product_distance(a, b)
    elif space == Space.Euclidean:
        expected = l2_square(a, b)
    else:
        raise NotImplementedError(f"Not sure how to calculate distance in tests for {space}!")

    assert (
        np.abs(actual - expected) < tolerance
    ), f"Expected {space.name} distance between {a} and {b} to be {expected}, but was {actual}"
