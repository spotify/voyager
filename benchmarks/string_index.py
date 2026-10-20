# Copyright 2026 Spotify AB
# Licensed under the Apache License, Version 2.0.
"""ASV benchmarks comparing native string identifiers with numeric labels.

Each timed insertion/update is run once per fresh setup to avoid timing a
progressively growing or repeatedly updated graph. Main/v2 skips StringIndex.
"""

from io import BytesIO

import numpy as np
import voyager


class Workload:
    def __init__(self, kind, size, dimensions=128, name_length=32, threads=1):
        if kind == "string" and not hasattr(voyager, "StringIndex"):
            raise NotImplementedError("This revision predates StringIndex")
        self.kind = kind
        self.size = size
        self.threads = threads
        rng = np.random.default_rng(1234)
        self.vectors = rng.uniform(-1, 1, (size, dimensions)).astype(np.float32)
        self.queries = rng.uniform(-1, 1, (256, dimensions)).astype(np.float32)
        self.update_ids = list(range(min(1000, size)))
        self.updates = rng.uniform(-1, 1, (len(self.update_ids), dimensions)).astype(np.float32)
        self.names = [f"item:{i:0{name_length - 5}d}" for i in range(size)]
        self.ids = list(range(size))
        self.update_names = [self.names[i] for i in self.update_ids]
        self.lookup_ids = rng.integers(0, size, size=1000).tolist()
        self.lookup_names = [self.names[i] for i in self.lookup_ids]
        self.index_class = voyager.StringIndex if kind == "string" else voyager.Index
        self.index = self.index_class(
            voyager.Space.Euclidean,
            dimensions,
            M=16,
            ef_construction=100,
            random_seed=4321,
            max_elements=size,
            storage_data_type=voyager.StorageDataType.Float32,
        )

    def insert(self):
        if self.kind == "string":
            self.index.add_items(self.names, self.vectors, num_threads=self.threads)
        else:
            self.index.add_items(self.vectors, self.ids, num_threads=self.threads)

    def update(self):
        if self.kind == "string":
            self.index.add_items(self.update_names, self.updates, num_threads=self.threads)
        else:
            self.index.add_items(self.updates, self.update_ids, num_threads=self.threads)

    def query(self):
        return self.index.query(self.queries, k=10, num_threads=self.threads, query_ef=100)

    def query_single(self):
        for vector in self.queries:
            self.index.query(vector, k=10, query_ef=100)

    def lookup(self):
        labels = self.lookup_names if self.kind == "string" else self.lookup_ids
        for label in labels:
            self.index.get_vector(label)

    def serialize(self):
        return self.index.as_bytes()

    def load(self):
        return self.index_class.load(BytesIO(self.serialized))


class StringIndexBuildSuite:
    params = (("numeric", "string"), (1000, 10000), (32, 128), (1, 4))
    param_names = ("identifiers", "num_elements", "name_bytes", "num_threads")
    number = 1
    repeat = (3, 3, 30.0)
    warmup_time = 0

    def setup(self, kind, size, name_length, threads):
        self.workload = Workload(kind, size, name_length=name_length, threads=threads)

    def time_insert(self, *_):
        self.workload.insert()


class StringIndexOperationsSuite(StringIndexBuildSuite):
    # Insertion has its own suite, with an empty index before each sample.
    time_insert = None

    def setup(self, kind, size, name_length, threads):
        super().setup(kind, size, name_length, threads)
        self.workload.insert()
        self.workload.serialized = self.workload.serialize()

    def time_update_existing(self, *_):
        self.workload.update()

    def time_query_batch(self, *_):
        self.workload.query()

    def time_query_single(self, *_):
        self.workload.query_single()

    def time_lookup(self, *_):
        self.workload.lookup()

    def time_save(self, *_):
        self.workload.serialize()

    def time_load(self, *_):
        self.workload.load()

    def track_serialized_bytes(self, *_):
        return len(self.workload.serialized)
