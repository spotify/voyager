package com.spotify.voyager.jni;

import com.spotify.voyager.jni.utils.JniLibExtractor;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * A Voyager index, providing storage of floating-point vectors and the ability to efficiently
 * search among those vectors.
 *
 * <p>A brief example of how to use {@code Index}:
 *
 * <pre>
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
public class Index implements Closeable {
  /** A C++ pointer! Don't futz with this from Java, or everything explodes. */
  private final long nativeHandle = 0;

  /**
   * The space, also known as distance metric, to use when searching.
   *
   * <p>SpaceType is a property of an Index, and cannot be changed after instantiation. Loading an
   * index with a different SpaceType than it was created with may result in nonsensical results
   * being returned.
   */
  public enum SpaceType {
    /**
     * Euclidean distance, also known as {@code L2} distance. Computed by taking the square root of
     * the sum of squared differences between each element of each vector.
     */
    Euclidean,

    /**
     * Inner (dot) product. Computed by taking the sum of the products of each element of each
     * vector.
     */
    InnerProduct,

    /**
     * Cosine distance; i.e. normalized dot product. Computed by taking the sum of the products of
     * each element of each vector, divided by the product of the magnitudes of each vector.
     */
    Cosine
  }

  /**
   * The datatype used to use when storing vectors on disk. Affects both precision and memory usage.
   */
  public enum StorageDataType {
    /**
     * An 8-bit floating point data type that expects all values to be on [-1, 1]. This data type
     * provides adequate precision for many use cases, but cuts down memory usage bu a factor of 4x
     * compared to Float32, while also increasing query speed.
     *
     * <p>Float8 provides 8 bits of resolution; i.e.: the distance between successive values is
     * 1/127, or 0.00787. For a variable-precision (i.e.: _actually_ floating point) representation,
     * use E4M3.
     */
    Float8,

    /** A 32-bit floating point ("Float") data type. The default. */
    Float32,

    /**
     * A custom 8-bit floating point data type with range [-448, 448] and variable precision. Use
     * this data type to get 4x less memory usage compared to Float32, but when the values of
     * vectors to be stored in an Index may exceed [-1, 1].
     *
     * <p>E4M3 uses a 4-bit exponent and 3-bit mantissa field, and was inspired by the paper "FP8
     * Formats for Deep Learning" by Micikevicus et al (arXiv:2209.05433).
     */
    E4M3
  }

  /**
   * A container for query results, returned by Index. Note that this class is instantiated from
   * C++, and as such, any changes to its location, visibility, or constructor will need to include
   * corresponding C++ changes.
   */
  public static class QueryResults {
    /** A list of item IDs ("labels"). */
    public final long[] labels;

    /** A list of distances from each item ID to the query vector for this query. */
    public final float[] distances;

    /**
     * Instantiates a new QueryResults object, provided two identical-length arrays of labels and
     * their corresponding distances. This method should probably not be used directly, as this
     * class is primarily used as a return type from the query method on Index.
     *
     * @throws IllegalArgumentException if the length of the labels and distances arrays vary
     */
    public QueryResults(long[] labels, float[] distances) {
      if (labels.length != distances.length) {
        throw new IllegalArgumentException("Labels and distances must have matching length!");
      }

      this.labels = labels;
      this.distances = distances;
    }

    public String toString() {
      return ("QueryResult(labels="
          + Arrays.toString(labels)
          + ", distances="
          + Arrays.toString(distances)
          + ")");
    }

    /**
     * Retrieve the list of item IDs ("labels") returned by this query. This array is sorted by
     * distance: the first item is the closest to the query vector, the second is second-closest,
     * and so on. The items in this array correspond 1:1 with the distances returned by
     * getDistances().
     */
    public long[] getLabels() {
      return labels;
    }

    /**
     * Retrieve the list of distances between query vectors and item vectors for the results of this
     * query. This array is sorted by distance: the first distance corresponds with the item the
     * closest to the query vector, the second is second-closest, and so on. The items in this array
     * correspond 1:1 with the labels returned by getLabels().
     */
    public float[] getDistances() {
      return distances;
    }
  }

  static {
    System.load(JniLibExtractor.extractBinaries("voyager"));
  }

  private Index() {}

  /**
   * Create a new {@link Index} that uses the given {@link Index.SpaceType} to store and compare
   * {@code numDimensions}-dimensional vectors.
   *
   * @param space the space type to use when storing and comparing vectors.
   * @param numDimensions the number of dimensions per vector.
   */
  public Index(SpaceType space, int numDimensions) {
    // Construct an index with default parameters:
    nativeConstructor(space, numDimensions, 16, 200, 1, 1, StorageDataType.Float32);
  }

  /**
   * Create a new {@link Index} that uses the given {@link Index.SpaceType} to store and compare
   * {@code numDimensions}-dimensional vectors.
   *
   * @param space The space type to use when storing and comparing vectors.
   * @param numDimensions The number of dimensions per vector.
   * @param indexM Controls the degree of interconnectedness between vectors. Higher values of
   *     {@code M} provide better recall (i.e. higher quality) but use more memory.
   * @param efConstruction Controls index quality, affecting the speed of {@code addItem} calls.
   *     Does not affect memory usage or size of the index.
   * @param randomSeed A random seed to use when initializing the index's internal data structures.
   * @param maxElements The maximum number of elements that this index can hold. This is a
   *     performance optimization; if the index contains this number of elements and {@link
   *     Index#addItem} or {@link Index#addItems} is called, the index's capacity will automatically
   *     expanded to fit the new elements. Setting {@code maxElements} in advance helps avoid these
   *     expensive resize operations if the number of elements to be added is already known.
   * @param storageDataType The datatype to use under-the-hood when storing vectors. Different data
   *     type options trade off precision for memory usage and query speed; see {@link
   *     Index.StorageDataType} for available data types.
   */
  public Index(
      SpaceType space,
      int numDimensions,
      long indexM,
      long efConstruction,
      long randomSeed,
      long maxElements,
      StorageDataType storageDataType) {
    nativeConstructor(
        space, numDimensions, indexM, efConstruction, randomSeed, maxElements, storageDataType);
  }

  /**
   * Load a Voyager index file and create a new {@link Index} initialized with the data in that
   * file.
   *
   * @param filename A filename to load.
   * @param space The {@link Index.SpaceType} to use when loading the index.
   * @param numDimensions The number of dimensions per vector.
   * @param storageDataType The {@link Index.StorageDataType} used by the index being loaded.
   * @return An {@link Index} whose contents have been initialized with the data provided by the
   *     file.
   * @throws RuntimeException if the index cannot be loaded from the file, or the file contains
   *     invalid data.
   */
  public static Index load(
      String filename, SpaceType space, int numDimensions, StorageDataType storageDataType) {
    Index index = new Index();
    index.nativeLoadFromFile(filename, space, numDimensions, storageDataType);
    return index;
  }

  /**
   * Interpret the contents of a {@code java.io.InputStream} as the contents of a Voyager index file
   * and create a new {@link Index} initialized with the data provided by that stream.
   *
   * @param inputStream A {@link java.io.InputStream} that will provide the contents of a Voyager
   *     index.
   * @param space The {@link Index.SpaceType} to use when loading the index.
   * @param numDimensions The number of dimensions per vector.
   * @param storageDataType The {@link Index.StorageDataType} used by the index being loaded.
   * @return An {@link Index} whose contents have been initialized with the data provided by the
   *     input stream.
   * @throws RuntimeException if the index cannot be loaded from the stream, or the stream contains
   *     invalid data.
   */
  public static Index load(
      InputStream inputStream,
      SpaceType space,
      int numDimensions,
      StorageDataType storageDataType) {
    Index index = new Index();
    index.nativeLoadFromInputStream(inputStream, space, numDimensions, storageDataType);
    return index;
  }

  /**
   * Close this {@link Index} and release any memory held by it. Note that this method must be
   * called to release the memory backing this {@link Index}; failing to do so may cause a memory
   * leak.
   *
   * <p>Any calls to methods after {@link Index#close()} is called will fail, as the underlying
   * native C++ object will have been deallocated.
   *
   * @throws IOException if the C++ destructor fails.
   */
  @Override
  public void close() throws IOException {
    nativeDestructor();
  }

  private native void nativeConstructor(
      SpaceType space,
      int numDimensions,
      long indexM,
      long efConstruction,
      long randomSeed,
      long maxElements,
      StorageDataType storageDataType);

  private native void nativeLoadFromFile(
      String filename, SpaceType space, int numDimensions, StorageDataType storageDataType);

  private native void nativeLoadFromInputStream(
      InputStream inputStream, SpaceType space, int numDimensions, StorageDataType storageDataType);

  private native void nativeDestructor();

  /**
   * Set the default EF ("query search depth") to use when {@link Index#query} is called.
   *
   * @param ef The new default EF value to use. This value can be overridden on a per-query basis at
   *     query time.
   */
  public native void setEf(long ef);

  /**
   * Get the default EF ("query search depth") that will be uses when {@link Index#query} is called.
   *
   * @return The current default EF value, used by the {@link Index} if no value is provided at
   *     query time.
   */
  public native int getEf();

  /**
   * Get the {@link Index.SpaceType} that this {@link Index} uses to store and compare vectors.
   *
   * @return The {@link Index.SpaceType} that is currently used by this {@link Index}.
   */
  public native SpaceType getSpace();

  /**
   * Get the number of dimensions used in this {@link Index}.
   *
   * @return The number of dimensions used by this {@link Index}, and which all vectors within this
   *     {@link Index} must have.
   */
  public native int getNumDimensions();

  /**
   * Set the default number of threads to use when adding multiple vectors in bulk, or when querying
   * for multiple vectors simultaneously.
   *
   * @param numThreads The default number of threads used for bulk-add or bulk-query methods if not
   *     overridden in each method call. Note that this affects the number of threads started for
   *     each method call - Voyager keeps no long-lived thread pool. For maximum efficiency, pass as
   *     much data as possible to each bulk-add or bulk-query method call to minimize overhead.
   */
  public native void setNumThreads(int numThreads);

  /**
   * Get the default number of threads used when adding multiple vectors in bulk oor when querying
   * for multiple vectors simultaneously.
   *
   * @return The default number of threads used for bulk-add or bulk-query methods if not overridden
   *     in each method call.
   */
  public native int getNumThreads();

  /**
   * Save this Index to a file at the provided filename. This file can be reloaded by using {@code
   * Index.load(...)}.
   *
   * @param pathToIndex The output filename to write to.
   */
  public native void saveIndex(String pathToIndex);

  /**
   * Save this Index to the provided output stream. The stream will not be closed automatically - be
   * sure to close the stream {@code saveIndex} has completed. The data written to the stream can be
   * reloaded by using {@code Index.load(...)}.
   *
   * @param outputStream The output stream to write to. This stream will not be closed
   *     automatically.
   */
  public native void saveIndex(OutputStream outputStream);

  /**
   * Add an item (a vector) to this {@link Index}. The item will automatically be given an
   * identifier equal to the return value of {@link Index#getNumElements()}.
   *
   * @param vector The vector to add to the index.
   * @throws RuntimeException If the provided vector does not contain exactly {@link
   *     Index#getNumDimensions()} dimensions.
   */
  public native void addItem(float[] vector);

  /**
   * Add an item (a vector) to this {@link Index} with the provided identifier.
   *
   * @param vector The vector to add to the index.
   * @param id The 64-bit integer denoting the identifier of this vector.
   * @throws RuntimeException If the provided vector does not contain exactly {@link
   *     Index#getNumDimensions()} dimensions.
   */
  public native void addItem(float[] vector, long id);

  /**
   * Add multiple items (vectors) to this {@link Index}.
   *
   * @param vectors The vectors to add to the index.
   * @param numThreads The number of threads to use when adding the provided vectors. If -1 (the
   *     default), the number of CPUs available on the current machine will be used.
   * @throws RuntimeException If any of the provided vectors do not contain exactly {@link
   *     Index#getNumDimensions()} dimensions.
   */
  public native void addItems(float[][] vectors, int numThreads);

  /**
   * Add multiple items (vectors) to this {@link Index}.
   *
   * @param vectors The vectors to add to the index.
   * @param ids The 64-bit identifiers that correspond with each of the provided vectors.
   * @param numThreads The number of threads to use when adding the provided vectors. If -1 (the
   *     default), the number of CPUs available on the current machine will be used. Note that this
   *     causes a temporary C++ thread pool to be used. Instead of calling {@link addItems} in a
   *     tight loop, consider passing more data to each {@link addItems} call instead to reduce
   *     overhead.
   * @throws RuntimeException If any of the provided vectors do not contain exactly {@link
   *     Index#getNumDimensions()} dimensions.
   * @throws RuntimeException If the list of IDs does not have the same length as the list of
   *     provided vectors.
   */
  public native void addItems(float[][] vectors, long[] ids, int numThreads);

  /**
   * Get the vector for the provided identifier.
   *
   * @param id The identifier whose vector will be fetched.
   * @return A {@link float} array representing the values of the vector.
   * @throws RuntimeException If the provided identifier is not present in the {@link Index}.
   */
  public native float[] getVector(long id);

  /**
   * Get the vectors for a provided array of identifiers.
   *
   * @param ids The identifiers whose vector will be fetched.
   * @return A nested {@link float} array representing the values of the vectors corresponding with
   *     each ID.
   * @throws RuntimeException If any of the provided identifiers are not present in the {@link
   *     Index}.
   */
  public native float[][] getVectors(long[] ids);

  /**
   * Get the list of identifiers currently stored by this index.
   *
   * @return a {@link long} array of identifiers.
   */
  public native long[] getIDs();

  /**
   * Query this {@link Index} for approximate nearest neighbors of a single query vector.
   *
   * @param queryVector A query vector to use for searching.
   * @param k The number of nearest neighbors to return.
   * @return A {@link QueryResults} object, containing the neighbors found that are (approximately)
   *     nearest to the query vector.
   * @throws RuntimeException if fewer than {@code k} results can be found in the index.
   */
  public QueryResults query(float[] queryVector, int k) {
    return query(queryVector, k, -1);
  }

  /**
   * Query this {@link Index} for approximate nearest neighbors of multiple query vectors.
   *
   * @param queryVectors The query vectors to use for searching.
   * @param k The number of nearest neighbors to return for each query vector
   * @param numThreads The number of threads to use when searching. If -1, all available CPU cores
   *     will be used. Note that passing a number of threads other than 1 will cause a temporary C++
   *     thread pool to be used. Instead of calling {@link query} in a tight loop, consider passing
   *     more data to each call instead to reduce overhead.
   * @return An array of {@link QueryResults} objects, each containing the neighbors found that are
   *     (approximately) nearest to the corresponding query vector. The returned list of {@link
   *     QueryResults} will contain the same number of elements as {@code queryVectors}.
   * @throws RuntimeException if fewer than {@code k} results can be found in the index for one or
   *     more queries.
   */
  public QueryResults[] query(float[][] queryVectors, int k, int numThreads) {
    return query(queryVectors, k, numThreads, -1);
  }

  /**
   * Query this {@link Index} for approximate nearest neighbors of a single query vector.
   *
   * @param queryVector A query vector to use for searching.
   * @param k The number of nearest neighbors to return.
   * @param queryEf The per-query "ef" value to use. Larger values produce more accurate results at
   *     the expense of query time.
   * @return A {@link QueryResults} object, containing the neighbors found that are (approximately)
   *     nearest to the query vector.
   * @throws RuntimeException if fewer than {@code k} results can be found in the index.
   */
  public native QueryResults query(float[] queryVector, int k, long queryEf);

  /**
   * Query this {@link Index} for approximate nearest neighbors of multiple query vectors.
   *
   * @param queryVectors The query vectors to use for searching.
   * @param k The number of nearest neighbors to return for each query vector
   * @param numThreads The number of threads to use when searching. If -1, all available CPU cores
   *     will be used. Note that passing a number of threads other than 1 will cause a temporary C++
   *     thread pool to be used. Instead of calling {@link query} in a tight loop, consider passing
   *     more data to each call instead to reduce overhead.
   * @param queryEf The per-query "ef" value to use. Larger values produce more accurate results at
   *     the expense of query time.
   * @return An array of {@link QueryResults} objects, each containing the neighbors found that are
   *     (approximately) nearest to the corresponding query vector. The returned list of {@link
   *     QueryResults} will contain the same number of elements as {@code queryVectors}.
   * @throws RuntimeException if fewer than {@code k} results can be found in the index for one or
   *     more queries.
   */
  public native QueryResults[] query(float[][] queryVectors, int k, int numThreads, long queryEf);

  /**
   * Mark an element of the index as deleted. Deleted elements will be skipped when querying, but
   * will still be present in the index.
   *
   * @param label The ID of the element to mark as deleted.
   * @throws RuntimeException If the provided identifier is not present in the {@link Index}.
   */
  public native void markDeleted(long label);

  /**
   * Un-mark an element of the index as deleted, making it available again.
   *
   * @param label The ID of the element to unmark as deleted.
   * @throws RuntimeException If the provided identifier is not present in the {@link Index}.
   */
  public native void unmarkDeleted(long label);

  /**
   * Change the maximum number of elements currently storable by this {@link Index}. This operation
   * reallocates the memory used by the index and can be quite slow, so it may be useful to set the
   * maximum number of elements in advance if that number is known.
   *
   * @param newSize The new number of maximum elements to resize this {@link Index} to.
   */
  public native void resizeIndex(long newSize);

  /**
   * Get the maximum number of elements currently storable by this {@link Index}. If more elements
   * are added than {@code getMaxElements()}, the index will be automatically (but slowly) resized.
   *
   * @return The number of elements (vectors) that are currently storable in this {@link Index}.
   */
  public native long getMaxElements();

  /**
   * Get the number of elements currently in this {@link Index}.
   *
   * @return The number of elements (vectors) in this {@link Index}. This count includes any deleted
   *     elements.
   */
  public native long getNumElements();

  /**
   * Get the EF Construction value used when adding new elements to this {@link Index}.
   *
   * @return The current EF Construction value (i.e.: the number of neighbors to search for when
   *     adding new elements).
   */
  public native long getEfConstruction();

  /**
   * Get the M value used when adding new elements to this {@link Index}.
   *
   * @return The current M value (i.e.: the number of links between adjacent vectors to create when
   *     adding elements).
   */
  public native long getM();
}
