# StringIndex performance measurements

This suite measures the public Python APIs of Voyager 3: native string identifiers
versus numeric labels on otherwise equivalent HNSW indexes. It does **not** compare
against Java's former `List<String>` wrapper, and it does not isolate C++ BiMap time
from Python conversion costs.

## Shared string storage

The current BiMap owns each string once in the name-to-label hash map. The
label-to-name map holds a pointer to that key. `std::unordered_map` preserves key
addresses across rehashes; copies rebuild the pointers against their own keys,
and moves transfer the owning nodes. Erasing an entry removes both directions.
The serialized representation is unchanged and contains string values, not pointers.

A macOS allocation probe isolates the BiMap from HNSW and Python. For 100,000
identifiers, it measures live allocator block bytes after insertion, including
size-class rounding:

| Name bytes | Original two-copy map | Shared-string map | Reduction |
| ---: | ---: | ---: | ---: |
| 32 | 20,871,168 B | 14,471,168 B | 30.7% |
| 128 | 43,271,168 B | 25,671,168 B | 40.7% |

For 32-byte names, the original two copies contain 6.4 MB of text but occupy
9.6 MB of string allocations. Hash nodes and bucket arrays account for another
11.27 MB. With shared storage, those figures fall to 3.2 MB of text, 4.8 MB of
string allocations, and 9.67 MB of hash nodes/buckets. Thus the memory difference
is not entirely string data: most remaining allocation bytes are map structure.
The original map's 20.87 MB of live allocations closely matches the 21.27 MB
extra RSS measured when loading the original string index, although RSS and
live heap allocations are different measurements.

Raw allocation results:
[original](results/bimap_allocations_original_macos.csv) and
[shared strings](results/bimap_allocations_shared_macos.csv).
The probe measures the original header from `2b5223e` and the shared-string header.

### End-to-end rerun

On the same machine and configuration below, three new paired samples of the
shared-string implementation produced these medians for 100,000 vectors with
32-byte names and one thread:

| Operation | Numeric Index | StringIndex | String / numeric |
| --- | ---: | ---: | ---: |
| Batch insertion | 13.460 s | 13.963 s | 1.04× |
| Batch query, per vector | 111.659 µs | 114.069 µs | 1.02× |
| Single query, per vector | 111.143 µs | 119.696 µs | 1.08× |
| Vector lookup, per identifier | 0.786 µs | 1.219 µs | 1.55× |
| Existing-name update, per vector | 985.125 µs | 1019.665 µs | 1.04× |
| Serialize to bytes | 10.142 ms | 14.599 ms | 1.44× |
| Load from bytes | 19.523 ms | 41.595 ms | 2.13× |
| Serialized file size | 63.007 MiB | 67.584 MiB | 1.07× |
| Resident-memory increase on file load | 80.266 MiB | 94.375 MiB | 1.18× |

Compared with the original run, string-index RSS falls from 100.562 to 94.375 MiB
(6.2% of the whole index). Extra RSS versus numeric identifiers falls from
20.281 to 14.109 MiB. At 10,000 items with 128-byte names it falls from 15.766 to
14.047 MiB (10.9% of the whole index). Serialized sizes are unchanged.

Query overhead remains broadly similar, but these separate three-sample runs do
not establish a latency improvement or rule out small regressions. Serialization
ratios vary substantially between runs. At 10,000 items with four threads, insert
and batch-query ratios remain 1.17× and 1.25×. The optimization reduces mapping
allocations; Python conversion and HNSW costs remain.

[Raw rerun samples](results/string_index_shared_names_macos_arm64.json) include
the SHA-256 of the optimized `BiMap.h`; the recorded commit is its parent because
the library was built from the working tree before committing these results.

## Original two-copy implementation: 100,000 vectors

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

## Original implementation: scaling and other cases

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

The JSON records settings, versions, the BiMap header hash, raw samples, medians,
and ratios. Override
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

The standalone allocation probe requires macOS and reports allocator bytes rather
than RSS. Input names are allocated before measurement; only live BiMap allocations
are counted. Its fixed 32- and 128-byte names use heap storage (the probe does not
measure small-string optimization). Reproduce the current mapping result with:

```sh
clang++ -std=c++17 -O2 -I cpp/src \
  benchmarks/scripts/measure_bimap_memory.cpp -o /tmp/measure-bimap-memory
/tmp/measure-bimap-memory
```

To measure the original mapping, extract `cpp/src/BiMap.h` from `2b5223e` into a
separate directory and put that directory before `cpp/src` on the include path.

## Runner maintainability

| Function | Complexity before | Complexity after |
| --- | ---: | ---: |
| `run_case` | 17 | 6 |

Extracted `measure_pair` (8) and `summarize` (4). The paired run validates neighbor,
distance, element-count, and updated-vector behavior while collecting samples.
