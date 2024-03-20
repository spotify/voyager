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

import com.spotify.voyager.jni.Index.SpaceType;
import com.spotify.voyager.jni.Index.StorageDataType;
import com.spotify.voyager.jni.utils.TinyJson;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Wrapper around com.spotify.voyager.jni.Index with a simplified interface which maps the index ID
 * to a provided String.
 *
 * <p>StringIndex can only accommodate up to 2^31 - 1 (2.1B) items, despite typical Voyager indices
 * allowing up to 2^63 - 1 (9e18) items.
 */
public class StringIndex implements Closeable {
  private static final String INDEX_FILE_NAME = "index.hnsw";
  private static final String NAMES_LIST_FILE_NAME = "names.json";
  private static final int DEFAULT_BUFFER_SIZE = 1024 * 1024 * 100;

  private final Index index;
  private final List<String> names;

  /**
   * Instantiate a new empty index with the specified space type and dimensionality
   *
   * @param spaceType Type of space and distance calculation used when determining distance between
   *     embeddings in the index, @see com.spotify.voyager.jni.Index.SpaceType
   * @param numDimensions Number of dimensions of each embedding stored in the underlying HNSW index
   */
  public StringIndex(SpaceType spaceType, int numDimensions) {
    this.index = new Index(spaceType, numDimensions);
    this.names = new ArrayList<>();
  }

  /**
   * Instantiate an empty index with the specified index parameters
   *
   * @param spaceType Type of space and distance calculation used when determining distance between
   *     embeddings in the index, @see com.spotify.voyager.jni.Index.SpaceType
   * @param numDimensions Number of dimensions of each embedding stored in the underlying HNSW index
   * @param indexM Number of connections made between nodes when inserting an element into the
   *     index. Increasing this value can improve recall at the expense of higher memory usage
   * @param efConstruction Search depth when inserting elements into the index. Increasing this
   *     value can improve recall (up to a point) at the cost of increased indexing time
   * @param randomSeed Random seed used during indexing
   * @param maxElements Initial size of the underlying HNSW index
   * @param storageDataType Type to store the embedding values as, @see
   *     com.spotify.voyager.jni.StorageDataType
   */
  public StringIndex(
      SpaceType spaceType,
      int numDimensions,
      long indexM,
      long efConstruction,
      long randomSeed,
      long maxElements,
      StorageDataType storageDataType) {
    this.index =
        new Index(
            spaceType,
            numDimensions,
            indexM,
            efConstruction,
            randomSeed,
            maxElements,
            storageDataType);
    this.names = new ArrayList<>();
  }

  private StringIndex(Index index, List<String> names) {
    this.index = index;
    this.names = names;
  }

  /**
   * Load a previously constructed index from the provided file location. It is important that the
   * dimensions, space type, and storage data type provided are the same that the index was
   * constructed with.
   *
   * @param indexFilename Filename of the underlying HNSW index
   * @param nameListFilename Filename of the JSON encoded names list
   * @param spaceType @see com.spotify.voyager.jni.Index.SpaceType
   * @param numDimensions Number of dimensions of each embedding stored in the underlying HNSW index
   * @param storageDataType @see com.spotify.voyager.jni.Index.StorageDataType
   * @return reference to the loaded StringIndex
   */
  public static StringIndex load(
      final String indexFilename,
      final String nameListFilename,
      final SpaceType spaceType,
      final int numDimensions,
      final StorageDataType storageDataType) {
    Index index = Index.load(indexFilename, spaceType, numDimensions, storageDataType);

    List<String> names;
    try {
      names = TinyJson.readStringList(Files.newInputStream(Paths.get(nameListFilename)));
    } catch (IOException e) {
      throw new RuntimeException("Error reading names list from: " + nameListFilename, e);
    }

    return new StringIndex(index, names);
  }

  /**
   * Load a previously constructed index from the provided input streams. It is important that the
   * dimensions, space type, and storage data type provided are the same that the index was
   * constructed with.
   *
   * @param indexInputStream input stream pointing to the underlying HNSW index
   * @param nameListInputStream input stream pointing to the JSON encoded names list
   * @param spaceType @see com.spotify.voyager.jni.Index.SpaceType
   * @param numDimensions Number of dimensions of each embedding stored in the underlying HNSW index
   * @param storageDataType @see com.spotify.voyager.jni.Index.StorageDataType
   * @return reference to the loaded StringIndex
   */
  public static StringIndex load(
      final InputStream indexInputStream,
      final InputStream nameListInputStream,
      final SpaceType spaceType,
      final int numDimensions,
      final StorageDataType storageDataType) {
    Index index =
        Index.load(
            new BufferedInputStream(indexInputStream, DEFAULT_BUFFER_SIZE),
            spaceType,
            numDimensions,
            storageDataType);
    List<String> names =
        TinyJson.readStringList(new BufferedInputStream(nameListInputStream, DEFAULT_BUFFER_SIZE));
    return new StringIndex(index, names);
  }

  /**
   * Load a previously constructed index from the provided file location. The space type,
   * dimensions, and storage data type are read from the file metadata.
   *
   * @param indexFilename Filename of the underlying HNSW index
   * @param nameListFilename Filename of the JSON encoded names list
   * @return reference to the loaded StringIndex
   */
  public static StringIndex load(final String indexFilename, final String nameListFilename) {
    Index index = Index.load(indexFilename);

    List<String> names;
    try {
      names = TinyJson.readStringList(Files.newInputStream(Paths.get(nameListFilename)));
    } catch (IOException e) {
      throw new RuntimeException("Error reading names list from: " + nameListFilename, e);
    }

    return new StringIndex(index, names);
  }

  /**
   * Load a previously constructed index from the provided input stream. The space type, dimensions,
   * and storage data type are read from the file metadata.
   *
   * @param indexInputStream input stream pointing to the underlying HNSW index
   * @param nameListInputStream input stream pointing to the JSON encoded names list
   * @return reference to the loaded StringIndex
   */
  public static StringIndex load(
      final InputStream indexInputStream, final InputStream nameListInputStream) {
    Index index = Index.load(new BufferedInputStream(indexInputStream, DEFAULT_BUFFER_SIZE));
    List<String> names =
        TinyJson.readStringList(new BufferedInputStream(nameListInputStream, DEFAULT_BUFFER_SIZE));
    return new StringIndex(index, names);
  }

  /**
   * Save the underlying index and JSON encoded name list to the provided output directory
   *
   * @param outputDirectory directory to output files to
   * @throws IOException when there is an error writing to JSON or saving to disk
   */
  public void saveIndex(final String outputDirectory) throws IOException {
    saveIndex(outputDirectory, INDEX_FILE_NAME, NAMES_LIST_FILE_NAME);
  }

  public void saveIndex(
      final String outputDirectory, final String indexFilename, final String nameListFilename)
      throws IOException {
    Path indexPath = Paths.get(outputDirectory, indexFilename);
    Path namesPath = Paths.get(outputDirectory, nameListFilename);
    try {
      this.index.saveIndex(indexPath.toString());

      final OutputStream outputStream = Files.newOutputStream(namesPath);
      TinyJson.writeStringList(this.names, outputStream);

      outputStream.flush();
      outputStream.close();
    } catch (Exception e) {
      Files.deleteIfExists(indexPath);
      Files.deleteIfExists(namesPath);
      throw e;
    }
  }

  /**
   * Save the underlying HNSW index and JSON encoded names list to the provided output streams
   *
   * @param indexOutputStream output stream pointing to the location to save the HNSW index
   * @param namesListOutputStream output stream pointing to the location to save the JSON names list
   * @throws IOException when there is an error writing to JSON or the output streams
   */
  public void saveIndex(OutputStream indexOutputStream, OutputStream namesListOutputStream)
      throws IOException {
    BufferedOutputStream outputStream =
        new BufferedOutputStream(indexOutputStream, 1024 * 1024 * 100);
    this.index.saveIndex(outputStream);
    TinyJson.writeStringList(this.names, namesListOutputStream);

    outputStream.flush();
    outputStream.close();
    namesListOutputStream.flush();
    namesListOutputStream.close();
  }

  public void addItem(String name, float[] vector) {
    int nextIndex = names.size();
    index.addItem(vector, nextIndex);
    names.add(name);
  }

  public void addItem(String name, List<Float> vector) {
    addItem(name, toPrimitive(vector));
  }

  public void addItems(Map<String, List<Float>> vectors) {
    int numVectors = vectors.size();

    List<String> newNames = new ArrayList<>(numVectors);
    float[][] primitiveVectors = new float[numVectors][index.getNumDimensions()];
    long[] labels = new long[numVectors];

    Iterator<Entry<String, List<Float>>> iterator = vectors.entrySet().iterator();
    for (int i = 0; i < numVectors; i++) {
      Entry<String, List<Float>> nextVector = iterator.next();
      newNames.add(nextVector.getKey());
      assignPrimitive(nextVector.getValue(), primitiveVectors[i]);
      labels[i] = names.size() + i;
    }

    names.addAll(newNames);
    index.addItems(primitiveVectors, labels, -1);
  }

  private float[] toPrimitive(List<Float> vector) {
    float[] vectorValues = new float[vector.size()];
    assignPrimitive(vector, vectorValues);
    return vectorValues;
  }

  private void assignPrimitive(List<Float> vector, float[] target) {
    for (int i = 0; i < target.length; i++) {
      target[i] = vector.get(i);
    }
  }

  /**
   * Find the nearest neighbors of the provided embedding.
   *
   * @param queryVector The vector to center the search around.
   * @param numNeighbors The number of neighbors to return. The number of results returned may be
   *     smaller than this value if the index does not contain enough items.
   * @param ef How many neighbors to explore during search when looking for nearest neighbors.
   *     Increasing this value can improve recall (up to a point) at the cost of increased search
   *     latency. The minimum value of this parameter is the requested number of neighbors, and the
   *     maximum value is the number of items in the index.
   * @return a QueryResults object, containing the names of the neighbors and each neighbor's
   *     distance from the query vector, sorted in ascending order of distance
   */
  public QueryResults query(float[] queryVector, int numNeighbors, int ef) {
    return convertResult(index.query(queryVector, numNeighbors, ef));
  }

  /**
   * Query for against multiple target vectors in parallel.
   *
   * @param queryVectors Array of query vectors to search around
   * @param numNeighbors Number of neighbors to get for each target
   * @param ef Search depth in the graph
   * @param numThreads Number of threads to use for the underlying index search. -1 uses all
   *     available CPU cores
   * @return Array of QueryResults, one for each target vector
   */
  public QueryResults[] query(float[][] queryVectors, int numNeighbors, int ef, int numThreads) {
    QueryResults[] results = new QueryResults[queryVectors.length];
    Index.QueryResults[] idxResults = index.query(queryVectors, numNeighbors, numThreads, ef);
    for (int i = 0; i < idxResults.length; i++) {
      results[i] = this.convertResult(idxResults[i]);
    }

    return results;
  }

  private QueryResults convertResult(Index.QueryResults idxResults) {
    int numResults = idxResults.distances.length;
    String[] resultNames = new String[numResults];
    float[] distances = new float[numResults];

    for (int i = 0; i < idxResults.getLabels().length; i++) {
      long indexId = idxResults.getLabels()[i];
      float dist = idxResults.getDistances()[i];
      if (indexId > Integer.MAX_VALUE || indexId < Integer.MIN_VALUE) {
        throw new ArrayIndexOutOfBoundsException(
            "Voyager index returned a label ("
                + indexId
                + ") which is out of range for StringIndex. "
                + "This index may not be compatible with Voyager's Java bindings, or the index file may be corrupt.");
      }
      String name = names.get((int) indexId);
      resultNames[i] = name;
      distances[i] = dist;
    }

    return new QueryResults(resultNames, distances);
  }

  @Override
  public void close() throws IOException {
    index.close();
  }

  /** A wrapper class for nearest neighbor query results. */
  public static class QueryResults {
    private final String[] names;
    private final float[] distances;

    public QueryResults(String[] names, float[] distances) {
      this.names = names;
      this.distances = distances;
    }

    public String[] getNames() {
      return this.names;
    }

    public float[] getDistances() {
      return this.distances;
    }

    public String getName(int index) {
      return this.names[index];
    }

    public float getDistance(int index) {
      return this.distances[index];
    }

    public int getNumResults() {
      return this.names.length;
    }

    @Override
    public String toString() {
      return "QueryResults{"
          + "names="
          + Arrays.toString(names)
          + ", distances="
          + Arrays.toString(distances)
          + '}';
    }
  }
}
