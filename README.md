![The word Voyager_in blue, with a multicoloured graphic illustrating an orbit to its left.](https://github.com/spotify/voyager/assets/213293/c99cd0e8-cd38-486f-bb61-15f74028ba52)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/spotify/voyager/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-on%20github.io-brightgreen)](https://spotify.github.io/voyager)
[![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)](https://pypi.org/project/voyager)
[![Apple Silicon support for macOS and Linux (Docker)](https://img.shields.io/badge/Apple%20Silicon-macOS%20and%20Linux-brightgreen)](https://pypi.org/project/voyager)
[![Test Badge](https://github.com/spotify/voyager/actions/workflows/all.yml/badge.svg)](https://github.com/spotify/voyager/actions/workflows/all.yml)
<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/voyager)](https://pypi.org/project/voyager) -->
<!-- [![PyPI - Wheel](https://img.shields.io/pypi/wheel/voyager)](https://pypi.org/project/voyager) -->
<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/voyager)](https://pypistats.org/packages/voyager) -->
<!-- [![GitHub Repo stars](https://img.shields.io/github/stars/spotify/voyager?style=social)](https://github.com/spotify/voyager/stargazers) -->

**_Voyager_** is a library for performing fast approximate nearest-neighbor searches on an in-memory collection of vectors.

Voyager features bindings to both Python and Java, with feature parity and index compatibility between both languages. It uses the HNSW algorithm, based on [the open-source `hnswlib` package](https://github.com/nmslib/hnswlib), with numerous features added for convenience and speed. Voyager is used extensively in production at Spotify, and is queried hundreds of millions of times per day to power numerous user-facing features.

Think of Voyager like [Sparkey](https://github.com/spotify/sparkey), but for vector/embedding data; or like [Annoy](https://github.com/spotify/annoy), but with [much higher recall](http://ann-benchmarks.com/). It got its name because it searches through (embedding) space(s), much like [the Voyager interstellar probes](https://en.wikipedia.org/wiki/Voyager_program) launched by NASA in 1977.

[![Python Documentation](https://img.shields.io/badge/Python%20Documentation-on%20github.io-brightgreen)](https://spotify.github.io/voyager/python)
[![Java Documentation](https://img.shields.io/badge/Java%20Documentation-on%20github.io-brightgreen)](https://spotify.github.io/voyager/java)

### Installation

#### Python

```shell
pip install voyager
```

#### Java

Add the following artifact to your `pom.xml`:
```xml
<dependency>
  <groupId>com.spotify</groupId>
  <artifactId>voyager</artifactId>
  <version>3.0.0</version>
</dependency>
```
You can find the latest version on [Voyager's Releases page](https://github.com/spotify/voyager/releases).

#### Scala

Add the following artifact to your `build.sbt`:
```sbt
"com.spotify" % "voyager" % "3.0.0"
```
You can find the latest version on [Voyager's Releases page](https://github.com/spotify/voyager/releases).


### String identifiers and Voyager 3

`StringIndex` stores string identifiers in a C++ bidirectional map. Adding an
existing name updates its vector; duplicate names within a batch use the last
vector. Names, including Unicode and embedded NUL characters, are stored with
the graph in a single binary file shared by C++, Java, and Python.

```python
from voyager import Space, StringIndex

index = StringIndex(Space.Euclidean, num_dimensions=2)
index.add_item("track:one", [0.0, 1.0])
index.add_item("track:one", [1.0, 0.0])  # Updates the existing item.
index.save("tracks.voy")
names, distances = StringIndex.load("tracks.voy").query([1.0, 0.0])
```

In Java, use `StringIndex.save("tracks.voy")` or
`saveIndex(OutputStream)` and `StringIndex.load("tracks.voy")` or
`load(InputStream)` for single-file persistence. The existing directory and
index-plus-JSON methods remain available for migration. Their binary output
also embeds the names and requires Voyager 3.

Voyager 3 reads Voyager 2 index files. To migrate a Java string index, load it
with the existing index-plus-JSON `StringIndex.load` overload, then save it as a
single file. In Python, use
`StringIndex.from_index(Index.load("index.hnsw"), json.load(names_file))`.
Legacy name lists must contain one unique name per numeric label, starting at
zero; duplicate names are rejected with an error rather than choosing a vector
silently.

All Voyager 3 saves, including numeric indexes and re-saved legacy indexes, use
file format **2** (Voyager 2 used file format **1**). Voyager 2 rejects these
files with its existing `unsupported version "0x2"` error and a message that a
newer Voyager library is required. Keep the original files if Voyager 2 clients
still need to read them.

### Compatibility

| OS        | Language | Version | x86_64 (Intel) | arm64 (ARM) |
|-----------|----------|---------|---------| --------|
| Linux     | Python   | 3.9     | ✅       | ✅       |
| Linux     | Python   | 3.10    | ✅       | ✅       |
| Linux     | Python   | 3.11    | ✅       | ✅       |
| Linux     | Python   | 3.12    | ✅       | ✅       |
| Linux     | Python   | 3.13    | ✅       | ✅       |
| Linux     | Java     | 8-16+   | ✅       | ✅      |
| macOS     | Python   | 3.9     | ✅       | ✅       |
| macOS     | Python   | 3.10    | ✅       | ✅       |
| macOS     | Python   | 3.11    | ✅       | ✅       |
| macOS     | Python   | 3.12    | ✅       | ✅       |
| macOS     | Python   | 3.13    | ✅       | ✅       |
| macOS     | Java     | 8-16+   | ✅       | ✅       |
| Windows   | Python   | 3.9     | ✅       | ❌       |
| Windows   | Python   | 3.10    | ✅       | ❌       |
| Windows   | Python   | 3.11    | ✅       | ❌       |
| Windows   | Python   | 3.12    | ✅       | ❌       |
| Windows   | Python   | 3.13    | ✅       | ❌       |
| Windows   | Java     | 8-16+   | ✅       | ❌       |


## Contributing

Contributions to `voyager` are welcomed!
See [CONTRIBUTING.md](https://github.com/spotify/voyager/blob/master/CONTRIBUTING.md) for details.


### License
Voyager is copyright 2022-2024 Spotify AB.

Voyager is licensed under the [Apache 2 License]([https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.apache.org/licenses/LICENSE-2.0)).
