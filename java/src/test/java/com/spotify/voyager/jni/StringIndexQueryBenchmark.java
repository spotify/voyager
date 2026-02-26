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
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmarks for StringIndex query performance.
 *
 * <p>Mirrors {@link IndexQueryBenchmark} but uses {@link StringIndex}, which wraps {@link Index}
 * with a string-name-to-numeric-ID mapping layer. Queries a pre-populated index of 4096 random
 * vectors with 256 dimensions, parameterized over space type and storage data type.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(2)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
public class StringIndexQueryBenchmark {

  @Param({"256"})
  public int numDimensions;

  @Param({"4096"})
  public int numElements;

  @Param({"Euclidean", "InnerProduct", "Cosine"})
  public String spaceType;

  @Param({"Float32", "Float8", "E4M3"})
  public String storageDataType;

  @Param({"24"})
  public int efConstruction;

  private static final int M = 20;
  private static final long RANDOM_SEED = 4321;
  private static final int DEFAULT_EF = -1;
  private static final int NAME_LENGTH = 32;
  private static final String NAME_PREFIX = "spotify:track:";

  private StringIndex index;
  private float[][] queryVectors;

  @Setup(Level.Trial)
  public void buildIndex() {
    Random rng = new Random(1234);
    boolean isFloat8 = "Float8".equals(storageDataType);

    float[][] inputData = new float[numElements][numDimensions];
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
    String[] itemNames = new String[numElements];
    for (int i = 0; i < numElements; i++) {
      itemNames[i] = NAME_PREFIX + randomAlphaString(nameRng, NAME_LENGTH);
    }

    Index.SpaceType space = Index.SpaceType.valueOf(spaceType);
    Index.StorageDataType storage = Index.StorageDataType.valueOf(storageDataType);
    index =
        new StringIndex(space, numDimensions, M, efConstruction, RANDOM_SEED, numElements, storage);
    for (int i = 0; i < numElements; i++) {
      index.addItem(itemNames[i], inputData[i]);
    }

    queryVectors = inputData;
  }

  @TearDown(Level.Trial)
  public void closeIndex() throws IOException {
    if (index != null) {
      index.close();
    }
  }

  /**
   * Queries the StringIndex for 1 nearest neighbor per vector.
   *
   * @param bh Blackhole to prevent dead-code elimination
   */
  @Benchmark
  public void queryK1(Blackhole bh) {
    for (float[] queryVector : queryVectors) {
      bh.consume(index.query(queryVector, 1, DEFAULT_EF));
    }
  }

  /**
   * Queries the StringIndex for 20 nearest neighbors per vector.
   *
   * @param bh Blackhole to prevent dead-code elimination
   */
  @Benchmark
  public void queryK20(Blackhole bh) {
    for (float[] queryVector : queryVectors) {
      bh.consume(index.query(queryVector, 20, DEFAULT_EF));
    }
  }

  private static String randomAlphaString(Random rng, int length) {
    char[] chars = new char[length];
    for (int i = 0; i < length; i++) {
      int val = rng.nextInt(52);
      if (val < 26) {
        chars[i] = (char) ('A' + val);
      } else {
        chars[i] = (char) ('a' + val - 26);
      }
    }
    return new String(chars);
  }
}
