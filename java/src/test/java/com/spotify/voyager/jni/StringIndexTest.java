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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import com.google.common.io.Resources;
import com.spotify.voyager.jni.Index.SpaceType;
import com.spotify.voyager.jni.Index.StorageDataType;
import com.spotify.voyager.jni.StringIndex.QueryResults;
import com.spotify.voyager.jni.TestUtils.Vector;
import com.spotify.voyager.jni.exception.RecallException;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
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

  private static final String TEMP_DIR_NAME = ".voyager-test-temp";
  private static final String EXPECTED_INDEX_FILE_NAME = "index.hnsw";
  private static final String EXPECTED_NAME_FILE_NAME = "names.json";
  private static final String EXPECTED_INDEX_V2_FILE_NAME = "index-v2.hnsw";
  private static final String EXPECTED_NAME_V2_FILE_NAME = "names-v2.json";

  @Test
  public void itFindsNeighbors() throws Exception {
    List<Vector> testVectors = TestUtils.getTestVectors();
    try (final StringIndex index =
        new StringIndex(
            SpaceType.Cosine,
            testVectors.get(0).vector.length,
            20,
            testVectors.size(),
            0,
            testVectors.size(),
            StorageDataType.E4M3)) {
      for (Vector v : testVectors) {
        index.addItem(v.name, v.vector);
      }

      List<CustomResult> results =
          RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, testVectors.size()));
      assertThat(results)
          .extracting(CustomResult::getName)
          .containsExactly("my-vector-78", "my-vector-93");
    }
  }

  @Test
  public void itFindsNeighborsMultipleTargets() throws Exception {
    List<Vector> testVectors = TestUtils.getTestVectors();
    int numDimensions = testVectors.get(0).vector.length;
    try (final StringIndex index =
        new StringIndex(
            SpaceType.Cosine,
            numDimensions,
            20,
            testVectors.size(),
            0,
            testVectors.size(),
            StorageDataType.E4M3)) {
      for (Vector v : testVectors) {
        index.addItem(v.name, v.vector);
      }

      float[][] targetVectors = TestUtils.randomVectors(() -> new Random(0), 2, numDimensions);

      List<List<CustomResult>> results =
          Arrays.stream(index.query(targetVectors, 2, 1, testVectors.size()))
              .map(RESULT_MAPPER)
              .collect(Collectors.toList());

      assertThat(results.stream().flatMap(List::stream))
          .extracting(CustomResult::getName)
          .containsExactly("my-vector-59", "my-vector-58", "my-vector-59", "my-vector-58");
    }
  }

  @Test
  public void itAddsItemsInBatch() throws Exception {
    List<Vector> testVectors = TestUtils.getTestVectors();
    try (final StringIndex index =
        new StringIndex(
            SpaceType.Cosine,
            testVectors.get(0).vector.length,
            16,
            testVectors.size(),
            0,
            testVectors.size(),
            StorageDataType.E4M3)) {

      Map<String, List<Float>> vectors =
          TestUtils.getTestVectors().stream()
              .collect(Collectors.toMap(vec -> vec.name, vec -> convert(vec.vector)));
      index.addItems(vectors);

      List<CustomResult> results =
          RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, testVectors.size()));
      assertThat(results)
          .extracting(CustomResult::getName)
          .containsExactly("my-vector-78", "my-vector-93");
    }
  }

  @Test
  public void itSavesToOutputStream() throws Exception {
    List<Vector> testVectors = TestUtils.getTestVectors();
    try (final StringIndex index =
        new StringIndex(
            SpaceType.Cosine,
            testVectors.get(0).vector.length,
            32,
            testVectors.size(),
            0,
            testVectors.size(),
            StorageDataType.E4M3)) {
      for (Vector v : testVectors) {
        index.addItem(v.name, v.vector);
      }

      File tempDir = new File(TEMP_DIR_NAME);
      if (!tempDir.mkdirs())
        throw new RuntimeException("Failed to make temporary directory in test!");

      try {
        index.saveIndex(
            Files.newOutputStream(Paths.get(TEMP_DIR_NAME, EXPECTED_INDEX_FILE_NAME)),
            Files.newOutputStream(Paths.get(TEMP_DIR_NAME, EXPECTED_NAME_FILE_NAME)));
        assertThat(Paths.get(TEMP_DIR_NAME, EXPECTED_INDEX_FILE_NAME).toFile()).exists();
        assertThat(Paths.get(TEMP_DIR_NAME, EXPECTED_NAME_FILE_NAME).toFile()).exists();

        final StringIndex reloadedIndex =
            StringIndex.load(
                Files.newInputStream(Paths.get(TEMP_DIR_NAME, EXPECTED_INDEX_FILE_NAME)),
                Files.newInputStream(Paths.get(TEMP_DIR_NAME, EXPECTED_NAME_FILE_NAME)),
                SpaceType.Cosine,
                testVectors.get(0).vector.length,
                StorageDataType.E4M3);
        List<CustomResult> results =
            RESULT_MAPPER.apply(reloadedIndex.query(TestUtils.TEST_VECTOR, 2, testVectors.size()));
        assertThat(results)
            .extracting(CustomResult::getName)
            .containsExactly("my-vector-78", "my-vector-93");
      } finally {
        FileUtils.deleteDirectory(tempDir);
      }
    }
  }

  @Test
  public void itSavesToFiles() throws Exception {
    try (final StringIndex index =
        new StringIndex(SpaceType.Cosine, 80, 32, 300, 0, 1, StorageDataType.E4M3)) {
      List<Vector> vectors = TestUtils.getTestVectors();

      for (Vector v : vectors) {
        index.addItem(v.name, v.vector);
      }

      File tempDir = new File(TEMP_DIR_NAME);
      if (!tempDir.mkdirs())
        throw new RuntimeException("Failed to make temporary directory in test!");

      try {
        index.saveIndex(TEMP_DIR_NAME);
        assertThat(Paths.get(TEMP_DIR_NAME, EXPECTED_INDEX_FILE_NAME)).exists();
        assertThat(Paths.get(TEMP_DIR_NAME, EXPECTED_NAME_FILE_NAME)).exists();
      } finally {
        FileUtils.deleteDirectory(tempDir);
      }
    }
  }

  @Test
  public void itLoadsFromInputStream() throws Exception {
    try (final StringIndex index =
        StringIndex.load(
            Resources.getResource(EXPECTED_INDEX_FILE_NAME).openStream(),
            Resources.getResource(EXPECTED_NAME_FILE_NAME).openStream(),
            SpaceType.Cosine,
            80,
            StorageDataType.E4M3)) {

      List<CustomResult> results = RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, 100));

      assertThat(results)
          .extracting(CustomResult::getName)
          .containsExactly("my-vector-78", "my-vector-93");
    }
  }

  @Test
  public void itLoadsFromInputStreamNoParams() throws Exception {
    try (final StringIndex index =
        StringIndex.load(
            Resources.getResource(EXPECTED_INDEX_V2_FILE_NAME).openStream(),
            Resources.getResource(EXPECTED_NAME_V2_FILE_NAME).openStream())) {

      List<CustomResult> results = RESULT_MAPPER.apply(index.query(TestUtils.TEST_VECTOR, 2, 100));

      assertThat(results)
          .extracting(CustomResult::getName)
          .containsExactly("my-vector-78", "my-vector-1");
    }
  }

  @Test
  public void itResizesIndex() throws Exception {
    List<Vector> testVectors = TestUtils.getTestVectors();
    try (final StringIndex index =
        new StringIndex(
            SpaceType.Cosine,
            testVectors.get(0).vector.length,
            20,
            testVectors.size(),
            0,
            testVectors.size(),
            StorageDataType.E4M3)) {
      for (Vector v : testVectors) {
        index.addItem(v.name, v.vector);
      }
      long currentSize = index.getMaxElements();
      index.resizeIndex(currentSize + 1);
      assertEquals(currentSize + 1, index.getMaxElements());
    }
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

  private List<Float> convert(float[] input) {
    List<Float> floats = new ArrayList<Float>(input.length);
    for (float f : input) floats.add(f);
    return floats;
  }

  @Test
  public void itUpdatesNamesAndSavesOneFile() throws Exception {
    String name = "東京🛰️\0café";
    java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
    try (StringIndex index = new StringIndex(SpaceType.Euclidean, 2)) {
      index.addItem(name, new float[] {0, 0});
      index.addItem(name, new float[] {1, 1});
      index.addItem("", new float[] {0, 0});
      assertThat(index.getNumElements()).isEqualTo(2);
      assertThat(index.getVector(name)).containsExactly(1, 1);
      assertThat(index.query(new float[] {1, 1}, 1, 10).getNames()).containsExactly(name);
      index.saveIndex(output);
    }
    try (StringIndex loaded =
        StringIndex.load(new java.io.ByteArrayInputStream(output.toByteArray()))) {
      assertThat(loaded.getVector(name)).containsExactly(1, 1);
      loaded.addItem(name, new float[] {0.5f, 0.5f});
      loaded.addItem("new", new float[] {1, 0});
      assertThat(loaded.getNumElements()).isEqualTo(3);
      assertThat(loaded.query(new float[] {0.5f, 0.5f}, 1, 10).getNames()).containsExactly(name);
      loaded.markDeleted(name);
      assertThat(loaded.query(new float[] {0.5f, 0.5f}, 2, 10).getNames()).doesNotContain(name);
      loaded.unmarkDeleted(name);
      assertThat(loaded.query(new float[] {0.5f, 0.5f}, 1, 10).getNames()).containsExactly(name);
    }
  }

  @Test
  public void nativeStringQueriesMatchNumericResults() throws Exception {
    int count = 64;
    Random random = new Random(42);
    float[][] vectors = TestUtils.randomVectors(() -> random, count, 16);
    String[] names = new String[count];
    for (int i = 0; i < count; i++) {
      names[i] = i % 2 == 0 ? "track-" + i : "東京🛰️\0-" + i;
    }
    for (SpaceType space : SpaceType.values()) {
      for (StorageDataType storage : StorageDataType.values()) {
        try (Index index = new Index(space, 16, 20, 100, 1, count, storage)) {
          index.enableStringIdentifiers();
          for (int i = 0; i < count; i++) {
            index.addStringItem(names[i], vectors[i]);
          }
          index.markDeleted(0);
          String[][] batchNames = new String[count][4];
          float[][] batchDistances = new float[count][4];
          index.queryStringsBatch(vectors, 4, 4, 100, batchNames, batchDistances);
          for (int i = 0; i < count; i++) {
            Index.QueryResults expected = index.query(vectors[i], 4, 100);
            String[] expectedNames =
                Arrays.stream(expected.getLabels())
                    .mapToObj(id -> names[(int) id])
                    .toArray(String[]::new);
            String[] actualNames = new String[4];
            float[] actualDistances = new float[4];
            index.queryStrings(vectors[i], actualNames, actualDistances, 100);
            assertThat(actualNames).containsExactly(expectedNames);
            assertThat(actualDistances).containsExactly(expected.getDistances());
            assertThat(batchNames[i]).containsExactly(expectedNames);
            assertThat(batchDistances[i]).containsExactly(expected.getDistances());
          }
        }
      }
    }
  }

  @Test
  public void stringQueriesPreserveExceptions() throws Exception {
    try (StringIndex index = new StringIndex(SpaceType.Euclidean, 2)) {
      index.addItem("only", new float[] {1, 0});
      assertThrows(RecallException.class, () -> index.query(new float[] {1, 0}, 2, 10));
      assertThrows(RecallException.class, () -> index.query(new float[][] {{1, 0}}, 2, 1, 10));
      assertThrows(RuntimeException.class, () -> index.query(new float[] {1}, 1, 10));
      assertThrows(RuntimeException.class, () -> index.query(new float[][] {{1}}, 1, 1, 10));
      assertThrows(RuntimeException.class, () -> index.query(new float[] {1, 0}, 2, 1));
      assertThat(index.query(new float[] {1, 0}, 1, 10).getNames()).containsExactly("only");
    }
  }

  @Test
  public void largeStringQueryBatchPreservesNames() throws Exception {
    String[] names = {"ascii", "nul\0end", "東京🛰️"};
    float[][] vectors = {{0, 0}, {1, 0}, {0, 1}};
    try (StringIndex index = new StringIndex(SpaceType.Euclidean, 2)) {
      for (int i = 0; i < names.length; i++) {
        index.addItem(names[i], vectors[i]);
      }
      float[][] queries = new float[2048][];
      for (int i = 0; i < queries.length; i++) {
        queries[i] = vectors[i % vectors.length];
      }
      QueryResults[] results = index.query(queries, 1, 4, 10);
      for (int i = 0; i < results.length; i++) {
        assertThat(results[i].getNames()).containsExactly(names[i % names.length]);
        assertThat(results[i].getDistances()).containsExactly(0);
      }
    }
  }

  @Test
  public void itPreservesIdentifierEncoding() throws Exception {
    String[] names = {
      "",
      "spotify:track:0123456789abcdefghijkL",
      "\u0001\u007f",
      "\0",
      "\0start",
      "middle\0end",
      "trailing\0",
      "\u0080\u07ff\u0800\ud7ff\ue000\uffff",
      "\ud800\udc00\udbff\udfff",
      "東京🛰️\0café",
      "unpaired-high-\ud800",
      "unpaired-low-\udc00",
      new String(new char[4096]).replace('\0', 'x')
    };
    for (String name : names) {
      // Preserve the Java UTF-8 encoder's replacement of unpaired surrogates.
      String expected = new String(name.getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8);
      java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
      try (StringIndex index = new StringIndex(SpaceType.Euclidean, 2)) {
        index.addItem(name, new float[] {1, 0});
        assertThat(index.query(new float[] {1, 0}, 1, 10).getNames()).containsExactly(expected);
        assertThat(index.getVector(name)).containsExactly(1, 0);
        index.saveIndex(output);
      }
      try (StringIndex loaded =
          StringIndex.load(new java.io.ByteArrayInputStream(output.toByteArray()))) {
        assertThat(loaded.query(new float[] {1, 0}, 1, 10).getNames()).containsExactly(expected);
      }
    }
  }

  @Test
  public void itUpdatesNamesInBatch() throws Exception {
    try (StringIndex index = new StringIndex(SpaceType.Euclidean, 2)) {
      index.addItem("a", new float[] {0, 0});
      java.util.Map<String, List<Float>> vectors = new java.util.LinkedHashMap<>();
      vectors.put("a", java.util.Arrays.asList(1f, 1f));
      vectors.put("b", java.util.Arrays.asList(0f, 0f));
      index.addItems(vectors);
      assertThat(index.getNumElements()).isEqualTo(2);
      assertThat(index.getVector("a")).containsExactly(1, 1);
    }
  }

  @Test
  public void itMigratesV2NamesIntoTheBinary() throws Exception {
    java.io.ByteArrayOutputStream output = new java.io.ByteArrayOutputStream();
    try (StringIndex legacy =
        StringIndex.load(
            Resources.getResource(EXPECTED_INDEX_V2_FILE_NAME).openStream(),
            Resources.getResource(EXPECTED_NAME_V2_FILE_NAME).openStream())) {
      legacy.saveIndex(output);
    }
    try (StringIndex loaded =
        StringIndex.load(new java.io.ByteArrayInputStream(output.toByteArray()))) {
      assertThat(loaded.query(TestUtils.TEST_VECTOR, 2, 100).getNames())
          .containsExactly("my-vector-78", "my-vector-1");
      long count = loaded.getNumElements();
      loaded.addItem("my-vector-78", TestUtils.TEST_VECTOR);
      assertThat(loaded.getNumElements()).isEqualTo(count);
    }
  }
}
