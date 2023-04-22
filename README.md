![Voyager Logo](https://ghe.spotify.net/storage/user/2565/files/d94529bc-dda4-444d-91aa-1d014bdaadf4)


**_Voyager_** is a library for performing extremely fast approximate nearest-neighbor searches on an in-memory index. Voyager features bindings to both Python and Java, with feature parity and index compatibility between both languages. It uses the HNSW algorithm, based on [the open-source `hnswlib`](https://github.com/nmslib/hnswlib), with numerous features added for Spotify-specific production use cases.

Think of Voyager like [Sparkey](https://github.com/spotify/sparkey), but for vector/embedding data; or like [Annoy](https://github.com/spotify/annoy), but with [much higher recall](http://ann-benchmarks.com/). It got its name because it searches through (embedding) space(s), much like [the Voyager interstellar probes](https://en.wikipedia.org/wiki/Voyager_program) launched by NASA in 1977.

[Documentation for Voyager](https://ghe.spotify.net/pages/vector-search-guild/voyager/) is currently available [on GitHub (GHE) Pages](https://ghe.spotify.net/pages/vector-search-guild/voyager/).

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

### Development

#### Release and Deployment

As _Voyager_ requires a native C++ binary to be available on each platform (macOS and Linux), the library must be built on a machine that can build macOS binaries. While _Voyager_ is still an internal Spotify package, we can't build macOS binaries on Tingle; instead, Tingle's master build just checks to see if the current version is available on PyPI and Artifactory, and fails the build if the current version is not available.

Deploying to Artifactory requires write permission from the #client-build squad - [check docs on Backstage](https://backstage.spotify.net/docs/default/component/artifactory/user/request-deploy-permissions/#local-credentials) or [talk to them on Slack](https://spotify.slack.com/archives/C02NEGV5F). [PyPI deployment instructions can be found on Google Docs](https://docs.google.com/document/d/1oiisPCPhS3fvtL07dn7hyXBf0DpDpgvVoVfOf8PRH2Q/edit#heading=h.dpeeuly3vi98).

The general practice is to:
1. Perform a release bump directly on master
   - To cut a release, update the version of whichever library(s) you're modifying:
     - Python version is managed in [setup.py](./python_bindings/setup.py)
     - Java version is managed in [pom.xml](./java_bindings/pom.xml) and in the [README](#java)
2. Build & deploy the new version.
   - You can build and deploy for macOS (Intel and Apple Silicon) and Linux (Intel) locally on an Apple Silicon machine:
    ```
    # deploy Java bindings
    cd java_bindings
    mvn deploy
    
    # deploy Python bindings:
    cd ../python_bindings
    rm -rf build dist
    python3 setup.py sdist bdist_wheel
    python3 -m twine upload -r spotify dist/*
    ```
3. Commit the version bump and push to master


#### Documentation

To rebuild documentation, use `javadoc`:
```
javadoc --source-path java_bindings/src/main/java com.spotify.voyager com.spotify.voyager.jni -d docs/java
```

Documentation is currently deployed to GitHub (GHE) Pages on the `master` branch.
