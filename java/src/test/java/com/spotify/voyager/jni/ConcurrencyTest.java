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
import static org.junit.Assert.fail;

import com.spotify.voyager.jni.Index.StorageDataType;
import java.io.IOException;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Ignore;
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

  @Test
  public void testIndexCanBeResizedWhileQuerying() {
    int numElements = 10_000;
    int sliceSize = 100;
    final Index index = new Index(Euclidean, 32, 5, 10, 1, 1, StorageDataType.Float32);
    float[][] inputData = TestUtils.randomQuantizedVectors(numElements, 32);

    index.addItems(Arrays.copyOfRange(inputData, 0, sliceSize), -1);

    AtomicInteger idx = new AtomicInteger(sliceSize);
    Thread t =
        new Thread(
            () -> {
              try {
                while (true) {
                  // add 100 elements every 10 ms
                  Thread.sleep(10);
                  int i = idx.getAndAdd(sliceSize);
                  if (i + sliceSize < numElements) {
                    float[][] toAdd = Arrays.copyOfRange(inputData, i, i + sliceSize);
                    index.addItems(toAdd, -1);
                  } else {
                    idx.set(numElements);
                    break;
                  }
                }

              } catch (Exception e) {
                System.out.println("Error adding item");
                throw new RuntimeException(e);
              }
            });
    t.start();

    Runnable queryIndex =
        () -> {
          while (true) {
            // query index for a random target every 1 ms
            try {
              index.query(TestUtils.randomQuantizedVectors(1, 32)[0], 1);
              Thread.sleep(1);
            } catch (Exception e) {
              e.printStackTrace();
            }
          }
        };

    Thread queryT = new Thread(queryIndex);
    Thread queryT1 = new Thread(queryIndex);
    Thread queryT2 = new Thread(queryIndex);
    queryT.start();
    queryT1.start();
    queryT2.start();

    while (idx.get() != numElements) {}
    // if we get here then we successfully added 10k items to the index without crashing
    assertTrue("Ran test with no crash", true);
  }

  @Test
  @Ignore("This test exposes a known bug, ignoring until a fix is put in")
  public void itCanAddItemsInParallel() {
    int numElements = 50_000;

    final Index index = new Index(Euclidean, 32, 5, 10, 1, 1, StorageDataType.Float32);
    float[][] inputData = TestUtils.randomQuantizedVectors(numElements, 32);

    index.addItem(inputData[0]);

    AtomicInteger idx = new AtomicInteger(1);
    AtomicBoolean running = new AtomicBoolean(true);
    AtomicReference<Optional<Throwable>> error = new AtomicReference<>(Optional.empty());

    Runnable addItem =
        () -> {
          try {
            while (true) {
              // add 1 item every millisecond
              Thread.sleep(1);
              int i = idx.getAndIncrement();
              if (i < numElements) {
                float[] toAdd = inputData[i];
                index.addItem(toAdd);
              } else {
                running.set(false);
              }
            }
          } catch (Exception e) {
            error.set(Optional.of(e));
            running.set(false);
          }
        };

    for (int i = 0; i < 5; i++) {
      Thread t = new Thread(addItem);
      t.start();
    }

    while (running.get()) {}
    // if we get here then we successfully added 50k items to the index without crashing
    if (error.get().isPresent()) {
      error.get().get().printStackTrace();
      fail("Error in test: " + error.get().get().getMessage());
    }
  }
}
