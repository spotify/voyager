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
 * JMH benchmarks for index query performance.
 *
 * <p>Mirrors the Python benchmark in benchmarks/index_query.py. Queries a pre-populated index of
 * 4096 random vectors with 256 dimensions, parameterized over space type, storage data type, and k.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(2)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
public class IndexQueryBenchmark {

  @Param({"Euclidean", "InnerProduct", "Cosine"})
  public String spaceType;

  @Param({"Float32", "Float8", "E4M3"})
  public String storageDataType;

  private static final int NUM_DIMENSIONS = 256;
  private static final int NUM_ELEMENTS = 4096;
  private static final int EF_CONSTRUCTION = 24;
  private static final int M = 20;
  private static final long RANDOM_SEED = 4321;

  private Index index;
  private float[][] queryVectors;

  @Setup(Level.Trial)
  public void buildIndex() {
    Random rng = new Random(1234);
    boolean isFloat8 = "Float8".equals(storageDataType);

    float[][] inputData = new float[NUM_ELEMENTS][NUM_DIMENSIONS];
    for (int i = 0; i < NUM_ELEMENTS; i++) {
      for (int j = 0; j < NUM_DIMENSIONS; j++) {
        float val = rng.nextFloat() * 2 - 1;
        if (isFloat8) {
          val = Math.round(val * 127f) / 127f;
        }
        inputData[i][j] = val;
      }
    }

    Index.SpaceType space = Index.SpaceType.valueOf(spaceType);
    Index.StorageDataType storage = Index.StorageDataType.valueOf(storageDataType);
    index =
        new Index(space, NUM_DIMENSIONS, M, EF_CONSTRUCTION, RANDOM_SEED, NUM_ELEMENTS, storage);
    index.addItems(inputData, 1);

    queryVectors = inputData;
  }

  @TearDown(Level.Trial)
  public void closeIndex() throws IOException {
    if (index != null) {
      index.close();
    }
  }

  @Benchmark
  public void queryK1(Blackhole bh) {
    for (float[] queryVector : queryVectors) {
      bh.consume(index.query(queryVector, 1));
    }
  }

  @Benchmark
  public void queryK20(Blackhole bh) {
    for (float[] queryVector : queryVectors) {
      bh.consume(index.query(queryVector, 20));
    }
  }
}
