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

import static com.spotify.voyager.jni.Index.SpaceType.Cosine;
import static com.spotify.voyager.jni.Index.SpaceType.Euclidean;
import static com.spotify.voyager.jni.Index.SpaceType.InnerProduct;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.spotify.voyager.jni.Index.StorageDataType;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.junit.Test;

public class IndexTest {
  // Allow up to 1% misses from approximate graph search, including unreachable nodes.
  private static final double MIN_SELF_QUERY_RECALL = 0.99;

  static final Map<Index.StorageDataType, Float> PRECISION_PER_DATA_TYPE = new HashMap<>();

  static {
    PRECISION_PER_DATA_TYPE.put(Index.StorageDataType.Float32, 0.00001f);
    PRECISION_PER_DATA_TYPE.put(Index.StorageDataType.Float8, 0.10f);
    PRECISION_PER_DATA_TYPE.put(Index.StorageDataType.E4M3, 0.20f);
  }

  @Test
  public void testEuclideanFloat32() throws Exception {
    runTestWith(Euclidean, 2000, Index.StorageDataType.Float32, false);
    runTestWith(Euclidean, 2000, Index.StorageDataType.Float32, true);
  }

  @Test
  public void testEuclideanFloat8() throws Exception {
    runTestWith(Euclidean, 2000, Index.StorageDataType.Float8, false);
    runTestWith(Euclidean, 2000, Index.StorageDataType.Float8, true);
  }

  @Test
  public void testEuclideanE4M3() throws Exception {
    runTestWith(Euclidean, 2000, Index.StorageDataType.E4M3, false);
    runTestWith(Euclidean, 2000, Index.StorageDataType.E4M3, true);
  }

  @Test
  public void testCosineFloat32() throws Exception {
    runTestWith(Cosine, 2000, Index.StorageDataType.Float32, false);
    runTestWith(Cosine, 2000, Index.StorageDataType.Float32, true);
  }

  @Test
  public void testCosineFloat8() throws Exception {
    runTestWith(Cosine, 2000, Index.StorageDataType.Float8, false);
    runTestWith(Cosine, 2000, Index.StorageDataType.Float8, true);
  }

  @Test
  public void testCosineE4M3() throws Exception {
    runTestWith(Cosine, 2000, Index.StorageDataType.E4M3, false);
    runTestWith(Cosine, 2000, Index.StorageDataType.E4M3, true);
  }

  @Test
  public void testInnerProductFloat32() throws Exception {
    runTestWith(InnerProduct, 2000, StorageDataType.Float32, false);
    runTestWith(InnerProduct, 2000, StorageDataType.Float32, true);
  }

  @Test
  public void testInnerProductFloat8() throws Exception {
    runTestWith(InnerProduct, 2000, StorageDataType.Float8, false);
    runTestWith(InnerProduct, 2000, StorageDataType.Float8, true);
  }

  @Test
  public void testInnerProductE4M3() throws Exception {
    runTestWith(InnerProduct, 2000, StorageDataType.E4M3, false);
    runTestWith(InnerProduct, 2000, StorageDataType.E4M3, true);
  }

  /**
   * One large test method with variable parameters, to replicate the parametrized tests we get "for
   * free" in Python with PyTest.
   */
  private void runTestWith(
      Index.SpaceType spaceType,
      int numElements,
      Index.StorageDataType storageDataType,
      boolean testSingleVectorMethods)
      throws Exception {
    try (Index index = new Index(spaceType, 32, 20, 2000, 1, 1, storageDataType)) {
      assertEquals(1, index.getMaxElements());

      float[][] inputData = TestUtils.randomQuantizedVectors(new Random(1234), numElements, 32);

      long[] ids = new long[inputData.length];
      for (int i = 0; i < inputData.length; i++) {
        ids[i] = i;
      }

      if (testSingleVectorMethods) {
        for (int i = 0; i < inputData.length; i++) {
          long addedId = index.addItem(inputData[i], ids[i]);
          assertEquals(ids[i], addedId);
        }
      } else {
        long[] addedIds = index.addItems(inputData, ids, -1);
        assertArrayEquals(ids, addedIds);
      }

      // Test property methods
      assertEquals(spaceType, index.getSpace());
      assertEquals(32, index.getNumDimensions());
      assertEquals(20, index.getM());
      assertEquals(2000, index.getEfConstruction());
      assertEquals(numElements, index.getMaxElements());

      long[] actualIds = index.getIDs();
      Arrays.sort(actualIds);

      assertArrayEquals(ids, actualIds);

      // Test getVector unless we're using a reduced-precision type:
      if (storageDataType != Index.StorageDataType.E4M3) {
        // ...and don't bother if we're using Cosine distance, which stores normalized
        // vectors instead of the original vectors:
        if (spaceType != Index.SpaceType.Cosine) {
          if (testSingleVectorMethods) {
            for (int i = 0; i < inputData.length; i++) {
              float[] fetchedVector = index.getVector(i);
              assertNotNull(fetchedVector);
              assertArrayEquals(fetchedVector, inputData[i], 0.01f);
            }
          } else {
            float[][] fetchedVectors = index.getVectors(ids);
            assertNotNull(fetchedVectors);

            for (int i = 0; i < inputData.length; i++) {
              assertNotNull(fetchedVectors[i]);
              assertArrayEquals(fetchedVectors[i], inputData[i], 0.01f);
            }
          }
        }
      }

      // Ensure getVector throws an exception on a missing ID:
      try {
        index.getVector(inputData.length + 1);
        assert (false);
      } catch (Exception e) {
        assert (true);
      }

      index.setEf(numElements);

      // Use the bulk-query interface with an explicit queryEf:
      for (long queryEf = 100; queryEf < numElements; queryEf *= 10) {
        checkQueryResults(
            index, storageDataType, inputData, index.query(inputData, 1, -1, queryEf));
      }
      // Use the single-query interface:
      for (long queryEf = 100; queryEf < numElements; queryEf *= 10) {
        Index.QueryResults[] singleResults = new Index.QueryResults[numElements];
        for (int i = 0; i < numElements; i++) {
          singleResults[i] = index.query(inputData[i], 1, queryEf);
        }
        checkQueryResults(index, storageDataType, inputData, singleResults);
      }
      // Use the batch-query interface with the index's default ef:
      for (long queryEf = 100; queryEf < numElements; queryEf *= 10) {
        checkQueryResults(index, storageDataType, inputData, index.query(inputData, 1, -1));
      }

      // Try to save the index to a byte string:
      ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
      index.saveIndex(outputStream);
      assertTrue(outputStream.size() > 0);

      byte[] indexAsBytes = index.asBytes();
      assertTrue(indexAsBytes.length > 0);
      assertTrue(Arrays.equals(indexAsBytes, outputStream.toByteArray()));

      // Recreate the index from the outputStream alone and ensure queries still work:
      try (Index reloadedIndex = Index.load(new ByteArrayInputStream(outputStream.toByteArray()))) {
        // TODO: Add ef to the set of metadata parameters that gets saved along with the index:
        // assertEquals(index.getEf(), reloadedIndex.getEf());

        reloadedIndex.setEf(index.getEf());
        checkQueryResults(
            reloadedIndex, storageDataType, inputData, reloadedIndex.query(inputData, 1, -1));
      }
      // Test markDeleted and unmarkDeleted
      index.markDeleted(0);

      // Try getting the vector out - we should not get it
      assertThrows(RuntimeException.class, () -> index.getVector(0));

      // Now let's unmark it
      index.unmarkDeleted(0);

      // And the resulting call should pass:
      assertNotNull(index.getVector(0));

      // Test resizing of index
      long currentSize = index.getMaxElements();
      index.resizeIndex(currentSize + 1);
      assertEquals(currentSize + 1, index.getMaxElements());
    }
  }

  private void checkQueryResults(
      Index index,
      StorageDataType storageDataType,
      float[][] inputData,
      Index.QueryResults[] results) {
    assertEquals(inputData.length, results.length);
    int matches = 0;
    boolean checkRecall =
        storageDataType != StorageDataType.E4M3 && index.getSpace() != InnerProduct;
    for (int i = 0; i < inputData.length; i++) {
      assertEquals(1, results[i].labels.length);
      assertEquals(1, results[i].distances.length);
      long label = results[i].labels[0];
      assertTrue("Returned label must belong to the index", label >= 0 && label < inputData.length);
      assertTrue("Returned distance must be finite", Float.isFinite(results[i].distances[0]));
      if (checkRecall) {
        if (label == i) {
          matches++;
        }
        // A search miss is allowed, but its distance must describe the returned vector.
        assertEquals(
            expectedDistance(inputData[i], inputData[(int) label], index, storageDataType),
            results[i].distances[0],
            PRECISION_PER_DATA_TYPE.get(storageDataType));
      }
    }
    if (checkRecall) {
      double recall = (double) matches / inputData.length;
      assertTrue(
          "Self-query recall " + recall + " < " + MIN_SELF_QUERY_RECALL,
          recall >= MIN_SELF_QUERY_RECALL);
    }
  }

  private float expectedDistance(
      float[] left, float[] right, Index index, StorageDataType storageDataType) {
    float[] a = queryRepresentation(left, index, storageDataType);
    float[] b = queryRepresentation(right, index, storageDataType);
    double result = 0;
    for (int i = 0; i < a.length; i++) {
      if (index.getSpace() == Euclidean) {
        double difference = a[i] - b[i];
        result += difference * difference;
      } else {
        result += (double) a[i] * b[i];
      }
    }
    return (float) (index.getSpace() == Euclidean ? result : 1 - result);
  }

  private float[] queryRepresentation(
      float[] vector, Index index, StorageDataType storageDataType) {
    float norm = 0;
    if (index.getSpace() == Cosine) {
      for (float value : vector) {
        norm += value * value;
      }
    }
    float scale = index.getSpace() == Cosine ? 1 / ((float) Math.sqrt(norm) + 1e-30f) : 1;
    float[] result = new float[vector.length];
    for (int i = 0; i < vector.length; i++) {
      float value = vector[i] * scale;
      result[i] = storageDataType == StorageDataType.Float8 ? (int) (value * 127) / 127.0f : value;
    }
    return result;
  }

  @Test
  public void testQueryConnectedIndex() throws Exception {
    float[][] vectors = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    long[] ids = {10, 20, 30, 40};
    for (Index.SpaceType space : Index.SpaceType.values()) {
      for (StorageDataType storage :
          new StorageDataType[] {StorageDataType.Float32, StorageDataType.Float8}) {
        // Four exactly representable vectors fit within every node's neighbor limit.
        try (Index index = new Index(space, 4, 20, 100, 1, 4, storage)) {
          index.addItems(vectors, ids, 1);
          Index.QueryResults[] results = index.query(vectors, 1, -1);
          for (int i = 0; i < vectors.length; i++) {
            assertArrayEquals(new long[] {ids[i]}, results[i].labels);
            assertArrayEquals(new float[] {0}, results[i].distances, 0);
            Index.QueryResults single = index.query(vectors[i], 1);
            assertArrayEquals(new long[] {ids[i]}, single.labels);
            assertArrayEquals(new float[] {0}, single.distances, 0);
            Index.QueryResults all = index.query(vectors[i], ids.length);
            assertEquals(ids[i], all.labels[0]);
            Arrays.sort(all.labels);
            assertArrayEquals(ids, all.labels);
            float distance = space == Euclidean ? 2 : 1;
            assertArrayEquals(new float[] {0, distance, distance, distance}, all.distances, 0);
          }
        }
      }
    }
  }
}
