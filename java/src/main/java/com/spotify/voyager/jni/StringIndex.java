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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Wrapper around com.spotify.voyager.jni.Index with a simplified interface which maps the HNSW
 * index ID to a provided String.
 */
public class StringIndex implements Closeable {
  private static final String INDEX_FILE_NAME = "index.hnsw";
  private static final String NAMES_LIST_FILE_NAME = "names.json";
  private final Index hnswIndex;
  private final List<String> names;

  /**
   * Instantiate a new empty index with the specified space type and dimensionality
   *
   * @param spaceType Type of space and distance calculation used when determining distance between
   *     embeddings in the index, @see com.spotify.voyager.jni.Index.SpaceType
   * @param numDimensions Number of dimensions of each embedding stored in the underlying HNSW index
   */
  public StringIndex(SpaceType spaceType, int numDimensions) {
    this.hnswIndex = new Index(spaceType, numDimensions);
    this.names = new ArrayList<>();
  }

  /**
   * Instantiate an empty index with the specified HNSW parameters
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
    this.hnswIndex =
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

  private StringIndex(Index hnswIndex, List<String> names) {
    this.hnswIndex = hnswIndex;
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
      String indexFilename,
      String nameListFilename,
      SpaceType spaceType,
      int numDimensions,
      StorageDataType storageDataType) {
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
      InputStream indexInputStream,
      InputStream nameListInputStream,
      SpaceType spaceType,
      int numDimensions,
      StorageDataType storageDataType) {

    // use buffered stream to keep read speeds high, reading up to 100MB at once
    BufferedInputStream inputStream = new BufferedInputStream(indexInputStream, 1024 * 1024 * 100);
    Index index = Index.load(inputStream, spaceType, numDimensions, storageDataType);
    List<String> names = TinyJson.readStringList(nameListInputStream);

    return new StringIndex(index, names);
  }

  /**
   * Save the underlying HNSW index and JSON encoded names list to the provided output directory
   *
   * @param outputDirectory directory to output files to
   * @throws IOException when there is an error writing to JSON or saving to disk
   */
  public void saveIndex(String outputDirectory) throws IOException {
    // TODO: the JNI index does not yet implement this method, this will fail for now
    String indexFilename = outputDirectory + "/" + INDEX_FILE_NAME;
    this.hnswIndex.saveIndex(indexFilename);

    String namesListFilename = outputDirectory + "/" + NAMES_LIST_FILE_NAME;
    final OutputStream outputStream = Files.newOutputStream(Paths.get(namesListFilename));
    TinyJson.writeStringList(this.names, outputStream);
    outputStream.flush();
    outputStream.close();
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
    this.hnswIndex.saveIndex(outputStream);
    TinyJson.writeStringList(this.names, namesListOutputStream);

    outputStream.flush();
    outputStream.close();
    namesListOutputStream.flush();
    namesListOutputStream.close();
  }

  public void addItem(String name, float[] vector) {
    int nextIndex = names.size();
    hnswIndex.addItem(vector, nextIndex);
    names.add(name);
  }

  public void addItem(String name, List<Float> vector) {
    float[] vectorVals = toPrimitive(vector);
    addItem(name, vectorVals);
  }

  public void addItems(Map<String, List<Float>> vectors) {
    int numVectors = vectors.size();

    List<String> newNames = new ArrayList(numVectors);
    float[][] primitiveVectors = new float[numVectors][];
    long[] labels = new long[numVectors];

    Iterator<Entry<String, List<Float>>> iterator = vectors.entrySet().iterator();
    for (int i = 0; i < numVectors; i++) {
      Entry<String, List<Float>> nextVector = iterator.next();
      newNames.add(nextVector.getKey());
      primitiveVectors[i] = toPrimitive(nextVector.getValue());
      labels[i] = names.size() + i;
    }

    names.addAll(newNames);
    hnswIndex.addItems(primitiveVectors, labels, -1);
  }

  private float[] toPrimitive(List<Float> vector) {
    float[] vectorVals = new float[vector.size()];
    for (int i = 0; i < vectorVals.length; i++) {
      vectorVals[i] = vector.get(i);
    }

    return vectorVals;
  }

  /**
   * Find the approximate nearest neighbors to the provided embedding
   *
   * @param queryVector the vector to search near
   * @param numNeighbors the number of requested neighbors
   * @param ef how many neighbors to explore during search when looking for nearest neighbors.
   *     Increasing this value can improve recall (up to a point) at the cost of increased search
   *     latency. Minimum value is the requested number of neighbors, maximum value is the number of
   *     items in the index.
   * @return a list of Result objects with their names and distances from the query vector, sorted
   *     by distance
   */
  public QueryResults query(float[] queryVector, int numNeighbors, int ef) {
    String[] resultNames = new String[numNeighbors];
    float[] distances = new float[numNeighbors];
    Index.QueryResults idxResults = hnswIndex.query(queryVector, numNeighbors, ef);
    for (int i = 0; i < idxResults.getLabels().length; i++) {
      long indexId = idxResults.getLabels()[i];
      float dist = idxResults.getDistances()[i];
      String name = names.get((int) indexId);
      resultNames[i] = name;
      distances[i] = dist;
    }

    return new QueryResults(resultNames, distances);
  }

  @Override
  public void close() throws IOException {
    hnswIndex.close();
  }

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

    @Override
    public String toString() {
      return "names: "
          + Arrays.toString(this.names)
          + "; distances: "
          + Arrays.toString(this.distances);
    }
  }
}
