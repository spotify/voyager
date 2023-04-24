/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 * --
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * -/-/-
 */

package com.spotify.voyager.jni;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.google.common.io.Resources;
import com.spotify.voyager.jni.Index.SpaceType;
import com.spotify.voyager.jni.Index.StorageDataType;
import com.spotify.voyager.jni.StringIndex.QueryResults;
import com.spotify.voyager.jni.TestUtils.ListVector;
import com.spotify.voyager.jni.TestUtils.Vector;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.commons.io.FileUtils;
import org.junit.Test;

public class StringIndexTest {
  private static final Function<QueryResults, List<CustomResult>> RESULT_MAPPER =
      qr -> {
        List<CustomResult> customResults = new ArrayList<>();
        for (int i = 0; i < qr.getNames().length; i++) {
          customResults.add(new CustomResult(qr.getNames()[i], qr.getDistances()[i]));
        }

        return customResults;
      };

  @Test
  public void itLoadsFromFileSystem() {}

  @Test
  public void itFindsNeighbors() throws Exception {
    StringIndex index = new StringIndex(SpaceType.Cosine, 80, 16, 200, 0, 50, StorageDataType.E4M3);
    List<Vector> vectors = TestUtils.getTestVectors();

    for (Vector v : vectors) {
      index.addItem(v.uri, v.vector);
    }

    List<CustomResult> results = RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, 2));
    assertThat(results)
        .extracting(CustomResult::getName)
        .containsExactly(
            "spotify:track:0Eqm7hD828cATBLUx2fJox", "spotify:track:0Eq2AzJQNwxf5FsLYBjitC");
    index.close();
  }

  @Test
  public void itFindsNeighborsUsingList() throws Exception {
    StringIndex index = new StringIndex(SpaceType.Cosine, 80, 16, 200, 0, 50, StorageDataType.E4M3);
    List<ListVector> vectors = TestUtils.getTestVectorsUsingList();

    for (ListVector v : vectors) {
      index.addItem(v.uri, v.vector);
    }

    List<CustomResult> results = RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, 2));
    assertThat(results)
        .extracting(CustomResult::getName)
        .containsExactly(
            "spotify:track:0Eqm7hD828cATBLUx2fJox", "spotify:track:0Eq2AzJQNwxf5FsLYBjitC");
    index.close();
  }

  @Test
  public void itAddsItemsInBatch() throws Exception {
    StringIndex index = new StringIndex(SpaceType.Cosine, 80, 16, 200, 0, 50, StorageDataType.E4M3);
    Map<String, List<Float>> vectors =
        TestUtils.getTestVectorsUsingList().stream()
            .collect(Collectors.toMap(vec -> vec.uri, vec -> vec.vector));

    index.addItems(vectors);

    List<CustomResult> results = RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, 2));
    assertThat(results)
        .extracting(CustomResult::getName)
        .containsExactly(
            "spotify:track:0Eqm7hD828cATBLUx2fJox", "spotify:track:0Eq2AzJQNwxf5FsLYBjitC");
    index.close();
  }

  @Test
  public void itSavesToOutputStream() throws Exception {
    StringIndex index = new StringIndex(SpaceType.Cosine, 80, 32, 300, 0, 1, StorageDataType.E4M3);
    List<Vector> vectors = TestUtils.getTestVectors();

    for (Vector v : vectors) {
      index.addItem(v.uri, v.vector);
    }

    File tempDir = new File("./temp");
    tempDir.mkdirs();

    index.saveIndex(
        Files.newOutputStream(Paths.get("./temp/index.hnsw")),
        Files.newOutputStream(Paths.get("./temp/names.json")));
    assertThat(new File("./temp/index.hnsw")).exists();
    assertThat(new File("./temp/names.json")).exists();

    FileUtils.deleteDirectory(tempDir);
    index.close();
  }

  @Test
  public void itSavesToFilesystem() throws Exception {
    StringIndex index = new StringIndex(SpaceType.Cosine, 80, 32, 300, 0, 1, StorageDataType.E4M3);
    List<Vector> vectors = TestUtils.getTestVectors();

    for (Vector v : vectors) {
      index.addItem(v.uri, v.vector);
    }

    File tempDir = new File("./temp2");
    tempDir.mkdirs();

    index.saveIndex("temp2");

    assertThat(new File("./temp2/index.hnsw")).exists();
    assertThat(new File("./temp2/names.json")).exists();

    FileUtils.deleteDirectory(tempDir);
    index.close();
  }

  @Test
  public void itLoadsFromInputStream() throws Exception {
    StringIndex index =
        StringIndex.load(
            Resources.getResource("index.hnsw").openStream(),
            Resources.getResource("names.json").openStream(),
            SpaceType.Cosine,
            80,
            StorageDataType.E4M3);

    List<CustomResult> results = RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, 2));

    assertThat(results)
        .extracting(CustomResult::getName)
        .containsExactly(
            "spotify:track:0Eqm7hD828cATBLUx2fJox", "spotify:track:0Eq2AzJQNwxf5FsLYBjitC");

    index.close();
  }

  @Test
  public void itVerifiesEfParameter() throws Exception {
    StringIndex index =
        StringIndex.load(
            Resources.getResource("index.hnsw").openStream(),
            Resources.getResource("names.json").openStream(),
            SpaceType.Cosine,
            80,
            StorageDataType.E4M3);

    assertThatThrownBy(() -> index.query(TestUtils.TEST_VECTOR, 2, 1))
        .isInstanceOf(RuntimeException.class)
        .hasMessage("queryEf must be equal to or greater than the requested number of neighbors");
  }

  public static class CustomResult {
    private final String name;
    private final float distance;

    public CustomResult(String name, float distance) {
      this.name = name;
      this.distance = distance;
    }

    public String getName() {
      return this.name;
    }

    public float getDistance() {
      return this.distance;
    }
  }
}
