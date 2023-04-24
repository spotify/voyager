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
        0.20396f,
        -0.193801f,
        0.010341f,
        -0.29898f,
        0.193092f,
        -0.269439f,
        0.010436f,
        -0.308213f,
        -0.007512f,
        0.331032f,
        -0.375417f,
        0.543712f,
        0.319875f,
        0.013286f,
        0.114932f,
        0.086575f,
        0.150309f,
        -0.650505f,
        0.561702f,
        0.059486f,
        -0.202891f,
        0.110894f,
        0.31372f,
        0.067179f,
        0.129272f,
        0.279586f,
        0.414332f,
        0.007588f,
        -0.323548f,
        -0.236852f,
        -0.126816f,
        -0.124074f,
        -0.163504f,
        -0.049097f,
        -0.1344f,
        0.00867f,
        0.359106f,
        0.069655f,
        -0.098123f,
        0.17702f,
        0.1217f,
        0.090834f,
        -0.073075f,
        0.486793f,
        -0.059276f,
        -0.069669f,
        -0.009353f,
        -0.333826f,
        0.09415f,
        0.107344f,
        -0.385969f,
        -0.110482f,
        0.242096f,
        -0.102329f,
        -0.074761f,
        -0.36569f,
        -0.173448f,
        -0.04389f,
        0.00386f,
        -0.353121f,
        -0.378965f,
        -0.209415f,
        0.085403f,
        0.006092f,
        0.240077f,
        0.542227f,
        0.171934f,
        0.322737f,
        -0.505381f,
        -0.195394f,
        -0.029179f,
        0.017475f,
        0.020546f,
        0.026908f,
        0.147108f,
        -0.206746f,
        0.264154f,
        0.213047f,
        -0.244341f,
        0.038912f
      };

  private static final Random random = new Random();
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
        Resources.getResource("sample-vectors.json"), new TypeReference<List<Vector>>() {});
  }

  public static List<ListVector> getTestVectorsUsingList() throws IOException {
    return OBJECT_MAPPER.readValue(
        Resources.getResource("sample-vectors.json"), new TypeReference<List<ListVector>>() {});
  }

  public static class Vector {
    public String uri;
    public float[] vector;

    @Override
    public String toString() {
      return "uri: " + uri + "; vectorVals: [" + Arrays.toString(vector) + "]";
    }
  }

  public static class ListVector {
    public String uri;
    public List<Float> vector;

    @Override
    public String toString() {
      return "uri: " + uri + "; vectorVals: [" + vector + "]";
    }
  }
}
