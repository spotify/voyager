# StringIndex performance measurements

This suite measures the public Python APIs of Voyager 3: native string identifiers
versus numeric labels on otherwise equivalent HNSW indexes. It does **not** compare
against Java's former `List<String>` wrapper, and it does not isolate C++ BiMap time
from Python conversion costs.

## Results: 100,000 vectors

Measured 2026-10-09 on an Apple M5 Max (18 CPU cores, 128 GB RAM), macOS 26.7.1,
Python 3.14.4, NumPy 2.5.3. Candidate implementation: `c4ab4ae`.
Each value is the median of three paired samples, with the order of numeric/string
measurements alternated between samples. Results below use one thread and 32-byte
ASCII names.

| Operation | Numeric Index | StringIndex | String / numeric |
| --- | ---: | ---: | ---: |
| Batch insertion (100,000 vectors) | 14.198 s | 15.097 s | 1.06× |
| Batch query, per vector | 116.882 µs | 120.518 µs | 1.03× |
| Single query, per vector | 119.102 µs | 126.048 µs | 1.06× |
| Vector lookup, per identifier | 0.714 µs | 1.123 µs | 1.57× |
| Existing-name update, per vector | 993.744 µs | 1001.519 µs | 1.01× |
| Serialize to bytes | 13.836 ms | 14.098 ms | 1.02× |
| Load from bytes | 19.458 ms | 43.458 ms | 2.23× |
| Serialized file size | 63.007 MiB | 67.584 MiB | 1.07× |
| Resident-memory increase on file load | 80.281 MiB | 100.562 MiB | 1.25× |

Insertion, query, and update overhead is modest in this configuration. Lookup is
about 57% slower but remains approximately one microsecond per vector. Loading
has the largest latency increase (2.23×), and loading a string index adds about
20.3 MiB more resident memory (roughly 213 extra bytes per item).

The file overhead is exactly `16 + UTF-8 name length` bytes per entry: 48 bytes
for a 32-byte name. This is a 7.3% increase for this graph/vector configuration.
The two in-memory hash maps also store hash-table entries and string allocations,
so memory overhead is substantially larger than the serialized overhead.

## Scaling and other cases

All latency columns below are ratios of string to numeric medians. Numeric and
string results, individual samples, and RSS samples are retained in
[`results/string_index_macos_arm64.json`](results/string_index_macos_arm64.json).

| Items | Name bytes | Threads | Insert | Batch query | Update | Load | RSS increase | File size |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 32 | 1 | 1.11× | 1.09× | 1.00× | 1.48× | 1.06× | 1.07× |
| 10,000 | 32 | 1 | 1.06× | 1.07× | 1.04× | 1.94× | 1.20× | 1.07× |
| 100,000 | 32 | 1 | 1.06× | 1.03× | 1.01× | 2.23× | 1.25× | 1.07× |
| 10,000 | 128 | 1 | 1.07× | 0.94× | 1.01× | 2.09× | 1.38× | 1.22× |
| 10,000 | 32 | 4 | 1.17× | 1.24× | 1.03× | 2.00× | 1.19× | 1.07× |

With four threads at 10,000 items, insertion overhead increases to 17% and batch
query overhead to 24%. The Python string batch path converts arrays into nested
C++ vectors, while the numeric path consumes NumPy arrays directly. That is a
plausible contributor to the difference, but these timings alone do not attribute
the overhead to particular functions.

## Method

- Deterministic, uniformly distributed vectors in `[-1, 1]`, 128 dimensions,
  Euclidean distance, Float32 storage, M=16, ef_construction=100, graph seed 4321.
- Capacity is preallocated. Insertion excludes constructor/allocation of capacity,
  input generation, and name generation. Both APIs receive contiguous float32
  NumPy arrays and explicit identifiers. Binding conversion is included.
- Queries use 256 independent vectors, k=10, query_ef=100. Single-query latency
  averages a loop of 256 API calls; batch-query latency divides one batch time by
  256. Lookup averages 1,000 calls over randomly selected stored identifiers.
- Updates change the first 1,000 existing identifiers to new vectors. Each sample
  starts with a fresh graph, so updates never time an already updated graph.
- Neighbor names and distances must match exactly between single-threaded builds.
  Updates must preserve the element count, and retrieved updated vectors are
  checked. Parallel builds may have different graph topology.
- Save and load timings use in-memory bytes and exclude filesystem latency. File
  size comes from those bytes. Save timing includes allocation/copying of bytes.
- Memory uses three fresh child processes per index type. Each warms binding
  initialization, records RSS, loads the saved file, and records RSS again while
  the index is alive. Reported values are median RSS increases, not peak RSS or
  exact allocated-byte counts; they include allocator and loader overhead.
- Index destruction is outside load timing. Lookup and batch query are warmed
  before timed reads. Single query and persistence are not separately warmed.

These are local synthetic microbenchmarks, not production latency guarantees.
Three samples are insufficient to establish small differences statistically;
for example, the long-name query ratio below 1× should not be read as an
improvement. Other dimensions, quantized storage, name distributions, index sizes,
CPU architectures, and Java JNI calls need separate measurements.

## Reproduce

Install the candidate library and dependencies, then run from the repository root:

```sh
python -m pip install ./python numpy psutil
python benchmarks/scripts/measure_string_index.py \
  --sizes 1000 10000 100000 --repeat 3 --extra-cases \
  --output /tmp/string-index-results.json
```

The JSON records settings, versions, raw samples, medians, and ratios. Override
`--sizes` and `--repeat` to trade runtime for scale or measurement precision.

The ASV suite in `string_index.py` adds insertion, update, single/batch query,
lookup, save/load, and file-size benchmarks to normal benchmark discovery. Each
mutation is timed once per setup; string cases skip revisions predating
`StringIndex`.

```sh
python -m pip install asv==0.6.3
asv check -E existing:python
asv machine --yes
asv run -E existing:python --bench '^string_index\.' --quick
```

ASV's quick mode validates execution; it is not the source of the table above.
Run without `--quick` for ASV's measured samples.

## Runner maintainability

| Function | Complexity before | Complexity after |
| --- | ---: | ---: |
| `run_case` | 17 | 6 |

Extracted `measure_pair` (8) and `summarize` (4). The paired run validates neighbor,
distance, element-count, and updated-vector behavior while collecting samples.
