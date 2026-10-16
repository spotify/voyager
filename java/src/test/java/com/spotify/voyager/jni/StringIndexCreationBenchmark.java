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

import java.io.IOException;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

/**
 * JMH benchmarks for StringIndex creation performance.
 *
 * <p>Mirrors {@link IndexCreationBenchmark} but uses {@link StringIndex}, which wraps {@link Index}
 * with a string-name-to-numeric-ID mapping layer. Measures the overhead of the StringIndex wrapper
 * during item insertion.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(2)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
public class StringIndexCreationBenchmark {

  @Param({"256"})
  public int numDimensions;

  @Param({"1024"})
  public int numElements;

  @Param({"Euclidean", "InnerProduct", "Cosine"})
  public String spaceType;

  @Param({"Float32", "Float8", "E4M3"})
  public String storageDataType;

  @Param({"24"})
  public int efConstruction;

  private static final int M = 20;
  private static final long RANDOM_SEED = 4321;
  private static final int NAME_LENGTH = 22;
  private static final String NAME_PREFIX = "spotify:track:";

  private float[][] inputData;
  private String[] itemNames;
  private StringIndex index;

  @Setup(Level.Trial)
  public void generateData() {
    Random rng = new Random(1234);
    inputData = new float[numElements][numDimensions];
    boolean isFloat8 = "Float8".equals(storageDataType);

    for (int i = 0; i < numElements; i++) {
      for (int j = 0; j < numDimensions; j++) {
        float val = rng.nextFloat() * 2 - 1;
        if (isFloat8) {
          val = Math.round(val * 127f) / 127f;
        }
        inputData[i][j] = val;
      }
    }

    Random nameRng = new Random(5678);
    itemNames = new String[numElements];
    for (int i = 0; i < numElements; i++) {
      itemNames[i] = NAME_PREFIX + randomBase62String(nameRng, NAME_LENGTH);
    }
  }

  @Setup(Level.Invocation)
  public void createFreshIndex() {
    Index.SpaceType space = Index.SpaceType.valueOf(spaceType);
    Index.StorageDataType storage = Index.StorageDataType.valueOf(storageDataType);
    index =
        new StringIndex(space, numDimensions, M, efConstruction, RANDOM_SEED, numElements, storage);
  }

  @TearDown(Level.Invocation)
  public void closeIndex() throws IOException {
    if (index != null) {
      index.close();
    }
  }

  /**
   * Adds items to the StringIndex one at a time using {@link StringIndex#addItem(String, float[])}.
   */
  @Benchmark
  public void addItems() {
    for (int i = 0; i < numElements; i++) {
      index.addItem(itemNames[i], inputData[i]);
    }
  }

  private static final String BASE62 =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  private static String randomBase62String(Random rng, int length) {
    char[] chars = new char[length];
    for (int i = 0; i < length; i++) {
      chars[i] = BASE62.charAt(rng.nextInt(BASE62.length()));
    }
    return new String(chars);
  }
}
