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

/**
 * Voyager is a Java and Python library that provides approximate nearest-neighbor search of vector
 * data. For most use cases, {@link com.spotify.voyager.jni.Index} will be the primary interface to
 * Voyager's functionality.
 *
 * <pre>
 *   // Import `Index`, the basic class that implements Voyager's functionality:
 *   import com.spotify.voyager.jni.Index;
 *
 *   // Create a new Index that compares 4-dimensional vectors via Euclidean distance:
 *   Index index = new Index(Index.SpaceType.Euclidean, 4);
 *
 *   // Add some data to the index:
 *   index.addItem(new float[]{1.0f, 2.0f, 3.0f, 4.0f});
 *   index.addItem(new float[]{2.0f, 3.0f, 4.0f, 5.0f});
 *
 *   // Query the index to return the `k` nearest neighbors of a given vector:
 *   Index.QueryResults results = index.query(new float[]{3.0f, 4.0f, 5.0f, 6.0f}, 1);
 *
 *   // Our query will return the 1th (second) item that was added, as it's the closest:
 *   assert(results.getLabels()[0] == 1);
 *
 *   // Serialize this index to use it again later:
 *   index.saveIndex("my_tiny_index.voy");
 * </pre>
 */
package com.spotify.voyager;
