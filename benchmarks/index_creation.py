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

import voyager


class IndexCreationSuite:

    repeat = (1, 10, 30.0)
    params = (
        [256],
        [1024],
        [voyager.Space.Euclidean, voyager.Space.InnerProduct, voyager.Space.Cosine],
        [voyager.StorageDataType.E4M3, voyager.StorageDataType.Float8, voyager.StorageDataType.Float32],
        [24],
    )
    param_names = ["num_dimensions", "num_elements", "space", "storage_data_type", "ef_construction"]

    def setup(
        self,
        num_dimensions: int,
        num_elements: int,
        space: voyager.Space,
        storage_data_type: voyager.StorageDataType,
        ef_construction: float,
    ):
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
            random_seed=4321,
        )

        self.input_data = input_data
        self.index = index

    def time_create(self, *_):
        self.index.add_items(self.input_data, num_threads=1)
