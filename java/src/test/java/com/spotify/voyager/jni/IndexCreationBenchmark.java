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
 * JMH benchmarks for index creation performance.
 *
 * <p>Mirrors the Python benchmark in benchmarks/index_creation.py. Measures the time to add 1024
 * random vectors of 256 dimensions to a fresh index, parameterized over space type and storage data
 * type.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(2)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
public class IndexCreationBenchmark {

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

  private float[][] inputData;
  private Index index;

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
  }

  @Setup(Level.Invocation)
  public void createFreshIndex() {
    Index.SpaceType space = Index.SpaceType.valueOf(spaceType);
    Index.StorageDataType storage = Index.StorageDataType.valueOf(storageDataType);
    index = new Index(space, numDimensions, M, efConstruction, RANDOM_SEED, numElements, storage);
  }

  @TearDown(Level.Invocation)
  public void closeIndex() throws IOException {
    if (index != null) {
      index.close();
    }
  }

  @Benchmark
  public void addItems() {
    index.addItems(inputData, 1);
  }
}
