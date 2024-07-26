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

from io import BytesIO
from itertools import product
from typing import Dict

import numpy as np

import voyager


class IndexQuerySuite:

    repeat = (1, 10, 30.0)
    params = (
        [256],
        [1024],
        [voyager.Space.Euclidean, voyager.Space.Cosine],
        [voyager.StorageDataType.E4M3, voyager.StorageDataType.Float8, voyager.StorageDataType.Float32],
        [24],
    )
    param_names = ["num_dimensions", "num_elements", "space", "storage_data_type", "ef_construction"]

    def setup_cache(self) -> Dict:

        param_combinations = product(*self.params)

        data = {}
        for param_combination in param_combinations:
            num_dimensions, num_elements, space, storage_data_type, ef_construction = param_combination

            generator = np.random.default_rng(seed=1234)
            input_data = generator.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1

            if storage_data_type == voyager.StorageDataType.Float8:
                input_data = np.round(input_data * 127) / 127

            index = voyager.Index(
                space=space,
                num_dimensions=num_dimensions,
                ef_construction=ef_construction,
                M=20,
                storage_data_type=storage_data_type,
            )
            index.add_items(input_data)

            data[param_combination] = (index.as_bytes(), input_data)

        return data

    def setup(
        self,
        cached_data: Dict,
        num_dimensions: int,
        num_elements: int,
        space: voyager.Space,
        storage_data_type: voyager.StorageDataType,
        ef_construction: float,
    ):
        index_as_bytes, self.input_data = cached_data[
            num_dimensions, num_elements, space, storage_data_type, ef_construction
        ]
        self.index = voyager.Index.load(BytesIO(index_as_bytes))

    def time_query_k_1(self, *_):
        self.index.query(self.input_data, k=1, num_threads=1)

    def time_query_k_20(self, *_):
        self.index.query(self.input_data, k=20, num_threads=1)

    def track_recall(self, *_):
        labels, _ = self.index.query(self.input_data, k=1, num_threads=1)
        matches = np.sum(labels[:, 0] == np.arange(len(self.input_data)))
        recall = matches / len(self.input_data)
        return recall
