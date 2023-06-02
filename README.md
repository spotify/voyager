[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/spotify/voyager/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-on%20github.io-brightgreen)](https://spotify.github.io/voyager)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/voyager)](https://pypi.org/project/voyager)
[![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)](https://pypi.org/project/voyager)
[![Apple Silicon support for macOS and Linux (Docker)](https://img.shields.io/badge/Apple%20Silicon-macOS%20and%20Linux-brightgreen)](https://pypi.org/project/voyager)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/voyager)](https://pypi.org/project/voyager)
[![Test Badge](https://github.com/spotify/voyager/actions/workflows/all.yml/badge.svg)](https://github.com/spotify/voyager/actions/workflows/all.yml)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/voyager)](https://pypistats.org/packages/voyager)
[![GitHub Repo stars](https://img.shields.io/github/stars/spotify/voyager?style=social)](https://github.com/spotify/voyager/stargazers)

**_Voyager_** is a library for performing extremely fast approximate nearest-neighbor searches on an in-memory index. Voyager features bindings to both Python and Java, with feature parity and index compatibility between both languages. It uses the HNSW algorithm, based on [the open-source `hnswlib`](https://github.com/nmslib/hnswlib), with numerous features added for Spotify-specific production use cases.

Think of Voyager like [Sparkey](https://github.com/spotify/sparkey), but for vector/embedding data; or like [Annoy](https://github.com/spotify/annoy), but with [much higher recall](http://ann-benchmarks.com/). It got its name because it searches through (embedding) space(s), much like [the Voyager interstellar probes](https://en.wikipedia.org/wiki/Voyager_program) launched by NASA in 1977.

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
  <version>[VERSION]</version>
</dependency>
```
you can find the latest version in our [POM](./java_bindings/pom.xml)
