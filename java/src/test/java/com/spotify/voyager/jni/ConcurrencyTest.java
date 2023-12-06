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

import static com.spotify.voyager.jni.Index.SpaceType.Euclidean;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.spotify.voyager.jni.Index.StorageDataType;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;

public class ConcurrencyTest {
  @Test
  public void testIndexCanBeClosedWhileBeingQueried() throws Exception {
    int numElements = 500000;
    final Index index = new Index(Euclidean, 32, 5, 10, 1, numElements, StorageDataType.Float32);
    float[][] inputData = TestUtils.randomQuantizedVectors(numElements, 32);

    long[] ids = new long[inputData.length];
    for (int i = 0; i < inputData.length; i++) {
      ids[i] = i;
    }

    index.addItems(inputData, ids, -1);

    final AtomicBoolean isQuerying = new AtomicBoolean(false);
    final long sleepMillis = 10;

    // Call close() in a separate thread, which
    // should call the native destructor immediately:
    Thread t =
        new Thread(
            () -> {
              try {
                while (!isQuerying.get()) {}
                Thread.sleep(sleepMillis);
                index.close();
              } catch (IOException | InterruptedException e) {
                throw new RuntimeException(e);
              }
            });

    t.start();

    // Ensure that a single query takes at least 10x longer than our sleep time:
    long before = System.currentTimeMillis();
    index.query(inputData[1], numElements / 2, numElements);
    long after = System.currentTimeMillis();
    assertTrue(
        "Query was too fast ("
            + (after - before)
            + "ms); adjust test parameters to make the query slower and to ensure race conditions can be adequately tested",
        (after - before) > sleepMillis * 10);

    // Querying should work until the index is closed, and
    // then throw an exception (rather than segfaulting or crashing):
    isQuerying.set(true);
    // Note: if this next line throws a RuntimeError, this test needs to be tuned to make the query
    // take longer.
    // The aim of this test is to have close() called in another thread while query() is executing.
    // query() should return successfully, but then future calls to query() should throw an
    // exception.
    index.query(inputData[1], numElements / 2, numElements);

    RuntimeException thrown =
        assertThrows(
            RuntimeException.class, () -> index.query(inputData[1], numElements / 2, numElements));

    assertTrue(
        "Expected exception message to contain 'has been closed', but was: " + thrown.getMessage(),
        thrown.getMessage().contains("has been closed"));
  }
}
