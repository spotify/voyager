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
