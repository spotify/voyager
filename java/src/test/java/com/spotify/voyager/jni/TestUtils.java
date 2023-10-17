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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.io.Resources;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class TestUtils {
  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
  public static final float[] TEST_VECTOR =
      new float[] {
        -0.28231781f, -0.09139011f, -0.41196093f, -0.2113353f, 0.66429873f,
        0.07280658f, -0.63757154f, -0.22123309f, -0.59409645f, 0.66991585f,
        0.39258104f, 0.75523863f, -0.12859088f, 0.7143025f, 0.12983862f,
        -0.57323257f, 0.83325341f, 0.97469045f, 0.61992435f, 0.96652926f,
        0.68755475f, 0.84308006f, 0.90231355f, -0.40115609f, -0.80228974f,
        0.18687592f, 0.47477202f, 0.96544036f, 0.83686646f, -0.5014816f,
        0.19722942f, 0.70090812f, 0.47494767f, -0.60224063f, -0.69739047f,
        0.21127793f, 0.10483031f, 0.95733437f, -0.75831836f, 0.01516447f,
        0.00345499f, 0.97712514f, 0.37446441f, -0.38833984f, 0.7659764f,
        -0.62995721f, -0.08325211f, -0.53427205f, -0.76960611f, 0.76637826f,
        -0.43707066f, 0.85080152f, 0.45881989f, 0.07228225f, 0.64695033f,
        0.60221812f, -0.50135875f, -0.42800623f, -0.93639702f, -0.03963584f,
        -0.2378898f, -0.01320721f, 0.96205932f, 0.34840291f, -0.94909593f,
        0.23807241f, -0.66140576f, 0.34590523f, -0.88397052f, 0.50043365f,
        0.55017077f, -0.04407538f, -0.69994084f, 0.55148539f, 0.75811802f,
        -0.78514799f, 0.77547708f, -0.49400604f, -0.88250679f, -0.45649873f
      };

  private static final Random random = new Random(0);

  /**
   * Generate a random floating-point vector whose values are on (-1, 1).
   *
   * @param numDimensions the number of dimensions to include in the vector
   * @return a randomly-initialized floating point vector with values on (-1, 1).
   */
  public static float[] randomVector(int numDimensions) {
    float[] vector = new float[numDimensions];
    for (int i = 0; i < numDimensions; i++) {
      vector[i] = random.nextFloat() * 2 - 1;
    }
    return vector;
  }

  /**
   * Generate a list of random floating-point vectors whose values are on (-1, 1).
   *
   * @param numElements the number of vectors in the resulting array
   * @param numDimensions the number of dimensions per vector
   * @return an array of float arrays, with numElements as the first dimension, and numDimensions as
   *     the second.
   */
  public static float[][] randomVectors(int numElements, int numDimensions) {
    float[][] vectors = new float[numElements][numDimensions];

    for (int i = 0; i < numElements; i++) {
      for (int j = 0; j < numDimensions; j++) {
        vectors[i][j] = random.nextFloat() * 2 - 1;
      }
    }

    return vectors;
  }

  /**
   * Generate a list of quantized random floating-point vectors whose values are on (-1, 1), rounded
   * to the nearest 0.1.
   *
   * @param numElements the number of vectors in the resulting array
   * @param numDimensions the number of dimensions per vector
   * @return an array of float arrays, with numElements as the first dimension, and numDimensions as
   *     the second.
   */
  public static float[][] randomQuantizedVectors(int numElements, int numDimensions) {
    float[][] vectors = new float[numElements][numDimensions];

    for (int i = 0; i < numElements; i++) {
      for (int j = 0; j < numDimensions; j++) {
        vectors[i][j] = (int) ((random.nextFloat() * 2 - 1) * 10.0f) / 10.0f;
      }
    }

    return vectors;
  }

  public static List<Vector> getTestVectors() throws IOException {
    return OBJECT_MAPPER.readValue(
        Resources.getResource("test-vectors.json"), new TypeReference<List<Vector>>() {});
  }

  public static class Vector {
    public String name;
    public float[] vector;

    @Override
    public String toString() {
      return "name: " + name + "; vectorVals: [" + Arrays.toString(vector) + "]";
    }
  }
}
