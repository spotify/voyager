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
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.junit.Test;

public class IndexTest {
  static Random random = new Random();

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

      float[][] inputData = TestUtils.randomQuantizedVectors(numElements, 32);

      long[] ids = new long[inputData.length];
      for (int i = 0; i < inputData.length; i++) {
        ids[i] = i;
      }

      if (testSingleVectorMethods) {
        for (int i = 0; i < inputData.length; i++) {
          index.addItem(inputData[i], ids[i]);
        }
      } else {
        index.addItems(inputData, ids, -1);
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

      final float expectedPrecision = PRECISION_PER_DATA_TYPE.get(storageDataType);

      // Use the bulk-query interface:
      Index.QueryResults[] results = new Index.QueryResults[0];
      for (long queryEf = 100; queryEf < numElements; queryEf *= 10) {
        results = index.query(inputData, /* k= */ 1, /* numThreads= */ -1, /* queryEf= */ queryEf);

        for (int i = 0; i < numElements; i++) {
          Index.QueryResults neighbor = results[i];

          assertEquals(1, neighbor.labels.length);
          assertEquals(1, neighbor.distances.length);

          // E4M3 is too low precision for us to confidently assume that querying with the
          // unquantized (fp32) vector will return the quantized vector as its NN
          if (storageDataType != Index.StorageDataType.E4M3) {
            long label = neighbor.labels[0];
            float distance = neighbor.distances[0];

            assertEquals(i, label);
            assertEquals(0.0f, distance, expectedPrecision);
          }
        }
      }

      // Use the single-query interface:
      for (long queryEf = 100; queryEf < numElements; queryEf *= 10) {
        for (int i = 0; i < numElements; i++) {
          final Index.QueryResults neighbor = index.query(inputData[i], 1, queryEf);

          assertEquals(1, neighbor.labels.length);
          assertEquals(1, neighbor.distances.length);

          // E4M3 is too low precision for us to confidently assume that querying with the
          // unquantized (fp32) vector will return the quantized vector as its NN
          if (storageDataType != Index.StorageDataType.E4M3) {
            assertEquals(i, neighbor.labels[0]);
            assertEquals(0.0f, neighbor.distances[0], expectedPrecision);
          }
        }
      }

      // Try to save the index to a byte string:
      ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
      index.saveIndex(outputStream);
      assertTrue(outputStream.size() > 0);

      // Recreate the index from the outputStream alone and ensure queries still work:
      try (Index reloadedIndex =
          Index.load(
              new ByteArrayInputStream(outputStream.toByteArray()),
              spaceType,
              32,
              storageDataType)) {
        final Index.QueryResults[] reloadedResults = reloadedIndex.query(inputData, 1, -1);
        for (int i = 0; i < numElements; i++) {
          Index.QueryResults neighbor = results[i];

          assertEquals(1, neighbor.labels.length);
          assertEquals(1, neighbor.distances.length);

          // E4M3 is too low precision for us to confidently assume that querying with the
          // unquantized (fp32) vector will return the quantized vector as its NN
          if (storageDataType != Index.StorageDataType.E4M3) {
            assertEquals(i, neighbor.labels[0]);
            assertEquals(0.0f, neighbor.distances[0], expectedPrecision);
          }
        }
      }
    }
  }
}
