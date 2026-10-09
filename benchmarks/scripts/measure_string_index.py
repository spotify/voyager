#!/usr/bin/env python3
# Copyright 2026 Spotify AB
# Licensed under the Apache License, Version 2.0.
"""Paired local measurements with raw samples and isolated load RSS.

Run from the repository root with the candidate Voyager installed:
  python benchmarks/scripts/measure_string_index.py --output /tmp/string-index.json
Requires numpy and psutil. Inputs/names are generated outside timed regions.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import voyager

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.string_index import Workload


def timed(operation):
    start = time.perf_counter_ns()
    value = operation()
    return (time.perf_counter_ns() - start) / 1e9, value


def memory_worker(kind, filename):
    """Fresh process: measure resident-memory increase while loading from disk."""
    index_class = voyager.StringIndex if kind == "string" else voyager.Index
    # Warm binding initialization without allocating the measured graph.
    warm = index_class(voyager.Space.Euclidean, 128, M=16, max_elements=1)
    del warm
    process = psutil.Process()
    gc.collect()
    before = process.memory_info().rss
    index = index_class.load(filename)
    after = process.memory_info().rss
    print(json.dumps({"rss_delta_bytes": after - before, "num_elements": index.num_elements}))


def measure_memory(kind, filename, repeat):
    samples = []
    for _ in range(repeat):
        output = subprocess.check_output([sys.executable, __file__, "--memory-worker", kind, str(filename)], text=True)
        samples.append(json.loads(output)["rss_delta_bytes"])
    return samples


def validate_pair(workloads):
    numeric, string = workloads["numeric"], workloads["string"]
    ids, distances = numeric.query()
    names, string_distances = string.query()
    expected = [[numeric.names[i] for i in row] for row in ids]
    assert names == expected, "String and numeric searches returned different neighbors"
    np.testing.assert_array_equal(distances, string_distances)
    for workload in workloads.values():
        assert workload.index.num_elements == workload.size


def measure_operations(workload):
    # Warm reads; keep graph mutation until after query/persistence measurements.
    workload.query()
    workload.lookup()
    result = {}
    for name, operation, divisor in [
        ("batch_query_us_per_vector", workload.query, len(workload.queries)),
        ("single_query_us_per_vector", workload.query_single, len(workload.queries)),
        ("lookup_us_per_vector", workload.lookup, len(workload.lookup_ids)),
    ]:
        result[name] = timed(operation)[0] * 1e6 / divisor
    seconds, workload.serialized = timed(workload.serialize)
    result["save_ms"] = seconds * 1000
    seconds, loaded = timed(workload.load)
    result["load_ms"] = seconds * 1000
    assert loaded.num_elements == workload.size
    del loaded
    result["file_bytes"] = len(workload.serialized)
    seconds, _ = timed(workload.update)
    result["update_us_per_vector"] = seconds * 1e6 / len(workload.update_ids)
    assert workload.index.num_elements == workload.size, "Update inserted duplicate nodes"
    return result


def measure_pair(size, name_length, threads, iteration, files):
    workloads = {kind: Workload(kind, size, name_length=name_length, threads=threads) for kind in files}
    order = list(files)
    if iteration % 2:
        order.reverse()
    inserts = {kind: timed(workloads[kind].insert)[0] * 1000 for kind in order}
    # With one construction thread, the two graphs and results must match.
    if threads == 1:
        validate_pair(workloads)
    measurements = {}
    for kind in order:
        workload = workloads[kind]
        metrics = measure_operations(workload)
        metrics["insert_ms"] = inserts[kind]
        measurements[kind] = metrics
        if iteration == 0:
            files[kind].write_bytes(workload.serialized)
        label = workload.update_names[-1] if kind == "string" else workload.update_ids[-1]
        np.testing.assert_array_equal(workload.index.get_vector(label), workload.updates[-1])
    return measurements


def summarize(samples, memory):
    summary = {}
    for kind, rows in samples.items():
        summary[kind] = {metric: statistics.median(row[metric] for row in rows) for metric in rows[0]}
        summary[kind]["load_rss_bytes"] = statistics.median(memory[kind])
    return summary


def run_case(size, name_length, threads, repeat, directory):
    samples = {"numeric": [], "string": []}
    files = {kind: directory / f"{kind}-{size}-{name_length}-{threads}.voy" for kind in samples}
    for iteration in range(repeat):
        measurements = measure_pair(size, name_length, threads, iteration, files)
        for kind, rows in samples.items():
            rows.append(measurements[kind])
        gc.collect()
        print(f"n={size} names={name_length} threads={threads}: sample {iteration + 1}/{repeat}", flush=True)
    memory = {kind: measure_memory(kind, files[kind], repeat) for kind in samples}
    summary = summarize(samples, memory)
    return {
        "num_elements": size,
        "name_bytes": name_length,
        "num_threads": threads,
        "samples": samples,
        "load_rss_samples_bytes": memory,
        "median": summary,
        "string_over_numeric": {
            metric: summary["string"][metric] / summary["numeric"][metric] for metric in summary["numeric"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/voyager-string-index.json"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--extra-cases", action="store_true", help="Also measure 128-byte names and four threads at 10k items"
    )
    parser.add_argument("--memory-worker", nargs=2, metavar=("KIND", "FILE"))
    args = parser.parse_args()
    if args.memory_worker:
        memory_worker(*args.memory_worker)
        return
    cases = [(size, 32, 1) for size in args.sizes]
    if args.extra_cases:
        cases.extend([(10000, 128, 1), (10000, 32, 4)])
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "voyager_version": list(voyager.version),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_threads": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "configuration": {
            "dimensions": 128,
            "M": 16,
            "ef_construction": 100,
            "query_ef": 100,
            "k": 10,
            "queries": 256,
            "storage": "Float32",
            "space": "Euclidean",
            "seed": 1234,
            "graph_seed": 4321,
            "repeat": args.repeat,
        },
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="voyager-benchmark-") as directory:
        for size, length, threads in cases:
            result["cases"].append(run_case(size, length, threads, args.repeat, Path(directory)))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Raw samples and medians: {args.output}")


if __name__ == "__main__":
    main()
