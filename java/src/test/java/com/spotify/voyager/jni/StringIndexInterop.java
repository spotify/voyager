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

/** Used by the optional Python/Java interoperability test. */
public class StringIndexInterop {
  public static void main(String[] args) throws Exception {
    String name = "東京🛰️\0café";
    if (args[0].equals("write")) {
      try (StringIndex index = new StringIndex(Index.SpaceType.Euclidean, 2)) {
        index.addItem(name, new float[] {0, 0});
        index.addItem("", new float[] {1, 0});
        index.save(args[1]);
      }
    } else {
      try (StringIndex index = StringIndex.load(args[1])) {
        if (index.getNumElements() != 3
            || !index.query(new float[] {0.5f, 0.5f}, 1, 10).getName(0).equals(name)
            || !index.query(new float[] {0, 1}, 1, 10).getName(0).equals("python")) {
          throw new AssertionError("Python-to-Java string index round trip failed");
        }
        index.addItem(name, new float[] {0, 0});
        if (index.getNumElements() != 3) throw new AssertionError("Update inserted a duplicate");
      }
    }
  }
}
