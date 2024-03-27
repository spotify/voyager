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

import com.google.common.io.Resources;
import com.spotify.voyager.jni.Index.SpaceType;
import com.spotify.voyager.jni.Index.StorageDataType;
import com.spotify.voyager.jni.StringIndex.QueryResults;
import com.spotify.voyager.jni.TestUtils.Vector;
import java.io.File;
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

  @Test
  public void itUpsertItem() throws Exception {
    List<Vector> testVectors = TestUtils.getTestVectors();
    try (final StringIndex index =
        new StringIndex(
            SpaceType.Cosine,
            testVectors.get(0).vector.length,
            20,
            2,
            0,
            2,
            StorageDataType.E4M3)) {
      Vector v1 = testVectors.get(0);
      Vector v2 = testVectors.get(1);
      // Add couple of vectors//
      index.addItem(v1.name, v1.vector);
      index.addItem(v2.name, v2.vector);
      // Again add the first one which should be an update not an insert with the same name
      index.addItem(v1.name, v1.vector);
      // We should only have 2 entries in it.
      assertEquals(2, index.getNumElements());
    }
  }

  @Test
  public void itUpsertItems() throws Exception {
    int testSize = 15;
    List<Vector> testVectors = TestUtils.getTestVectors();
    try (final StringIndex index =
        new StringIndex(
            SpaceType.Cosine,
            testVectors.get(0).vector.length,
            20,
            testSize,
            0,
            testSize,
            StorageDataType.E4M3)) {
      Map<String, List<Float>> vectors =
          testVectors.stream()
              .limit(testSize)
              .collect(Collectors.toMap(vec -> vec.name, vec -> convert(vec.vector)));
      index.addItems(vectors);
      assertEquals(vectors.size(), index.getNumElements());
      // If we add it again number of elements  should stay the same//
      index.addItems(vectors);
      assertEquals(vectors.size(), index.getNumElements());
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
}
