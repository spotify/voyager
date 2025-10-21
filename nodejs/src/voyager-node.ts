// @ts-ignore
import gyp from "node-gyp-build";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __root = dirname(dirname(__filename));

// Load the native module
const native = gyp(__root);

// The method used to calculate the distances between vectors.
export enum Space {
  // Euclidean distance (L2 distance)
  Euclidean = 0,
  // Inner product distance
  InnerProduct = 1,
  // Cosine distance (normalized inner product)
  Cosine = 2,
}

// The data type used to store vectors in memory and on disk.
export enum StorageDataType {
  // 8-bit fixed-point decimal values. All values must be within [-1, 1.00787402]
  Float8 = 16,
  // 32-bit floating point (default)
  Float32 = 32,
  // 8-bit floating point with range [-448, 448]
  E4M3 = 48,
}

// Options for creating a new Index
export interface IndexOptions {
  // The space/distance metric to use
  space: Space;
  // The number of dimensions in each vector
  numDimensions: number;
  // The number of connections between nodes (default: 12)
  M?: number;
  // The number of vectors to search when inserting (default: 200)
  efConstruction?: number;
  // Random seed for reproducible index creation (default: 1)
  randomSeed?: number;
  // Initial maximum number of elements (default: 1)
  maxElements?: number;
  // Storage data type (default: Float32)
  storageDataType?: StorageDataType;
}

// Options for loading an index from disk
export interface LoadOptions {
  // The space/distance metric (for legacy indices without metadata)
  space?: Space;
  // The number of dimensions (for legacy indices without metadata)
  numDimensions?: number;
  // Storage data type (for legacy indices without metadata)
  storageDataType?: StorageDataType;
}

// Result from querying a single vector
export interface QueryResult {
  // Array of neighbor IDs
  neighbors: number[];
  // Array of distances
  distances: number[];
}

// Result from querying multiple vectors
export interface QueryResults {
  // Array of neighbor ID arrays (one per query)
  neighbors: number[][];
  // Array of distance arrays (one per query)
  distances: number[][];
}

/** A nearest-neighbor search index containing vector data.
 * Think of a Voyager Index as a Map<number, number[]> where you can
 * efficiently find the k nearest keys to a query vector.
 */
export class Index {
  private _index: any;

  // Create a new Index with the specified options
  constructor(options: IndexOptions) {
    this._index = new native.Index(options);
  }

  /** Add a single vector to the index
    * @param vector - The vector to add (array of numbers)
    * @param id - Optional ID to assign
    (auto-generated if not provided)
    * @returns The ID assigned to this vector 
    */
  addItem(vector: number[], id?: number): number {
    return this._index.addItem(vector, id);
  }

  /** Add multiple vectors to the index simultaneously
   * @param vectors - Array of vectors to add
   * @param ids - Optional array of IDs (must match vectors length if palovided)
   * @param numThreads - Number of threads to use (-1 for auto)
   * @returns Array of IDs assigned to the vectors
   */
  addItems(vectors: number[][], ids?: number[], numThreads?: number): number[] {
    return this._index.addItems(vectors, ids, numThreads);
  }
  /** Query the index for nearest neighbors of a single vector
   * @param vector - Vector to query
   * @param k - Number of neighbors to return (default: 1)
   * @param numThreads - Number of threads for parallel queries (-1 for auto)
   * @param queryEf - Search depth for this query (-1 to use default ef)
   * @returns Object containing neighbors and distances arrays
   */
  query(
    vectors: number[],
    k?: number,
    numThreads?: number,
    queryEf?: number
  ): QueryResult;

  /** Query the index for nearest neighbors of multiple vectors
   * @param vectors - Array of vectors to query
   * @param k - Number of neighbors to return (default: 1)
   * @param numThreads - Number of threads for parallel queries (-1 for auto)
   * @param queryEf - Search depth for this query (-1 to use default ef)
   * @returns Object containing arrays of neighbors and distances
   */
  query(
    vectors: number[][],
    k?: number,
    numThreads?: number,
    queryEf?: number
  ): QueryResults;

  query(
    vectors: number[] | number[][],
    k?: number,
    numThreads?: number,
    queryEf?: number
  ): QueryResult | QueryResults {
    return this._index.query(vectors, k, numThreads, queryEf);
  }

  /** Get the vector stored at the given ID
   * @param id - The ID to retrieve
   * @returns The vector as an array of numbers
   */
  getVector(id: number): number[] {
    return this._index.getVector(id);
  }

  /** Get multiple vectors by their IDs
   * @param ids - Array of IDs to retrieve
   * @returns Array of vectors
   */
  getVectors(ids: number[]): number[][] {
    return this._index.getVectors(ids);
  }

  /** Mark an ID as deleted (will not appear in query results)
   * @param id - The ID to mark as deleted
   */
  markDeleted(id: number): void {
    this._index.markDeleted(id);
  }

  /** Unmark an ID as deleted (will appear in query results again)
   * @param id - The ID to unmark
   */
  unmarkDeleted(id: number): void {
    this._index.unmarkDeleted(id);
  }

  /** Resize the index to accommodate more elements
   * @param newSize - New maximum number of elements
   */
  resize(newSize: number): void {
    this._index.resize(newSize);
  }

  /** Save the index to a file
   * @param filePath - Path where the index should be saved
   */
  saveIndex(filePath: string): void {
    this._index.saveIndex(filePath);
  }

  /** Load an index from a file
   * @param filePath - Path to the index file
   * @param options - Optional parameters for loading legacy indices
   * @returns A new Index instance
   */
  static loadIndex(filePath: string, options?: LoadOptions): Index {
    const nativeIndex = native.Index.loadIndex(filePath, options);
    const index = Object.create(Index.prototype);
    index._index = nativeIndex;
    return index;
  }

  /** Get the distance between two vectors
   * @param a - First vector
   * @param b - Second vector
   * @returns The distance between the vectors
   */
  getDistance(a: number[], b: number[]): number {
    return this._index.getDistance(a, b);
  }

  /** Serialize the index to a Buffer
   * @returns Buffer containing the serialized index
   */
  toBuffer(): Buffer {
    return this._index.toBuffer();
  }

  /** Load an index from a Buffer
   * @param buffer - Buffer containing the serialized index
   * @param options - Optional parameters for loading legacy indices
   * @returns A new Index instance
   */
  static fromBuffer(buffer: Buffer, options?: LoadOptions): Index {
    const nativeIndex = native.Index.fromBuffer(buffer, options);
    const index = Object.create(Index.prototype);
    index._index = nativeIndex;
    return index;
  }

  /** Check if an ID exists in the index
   * @param id - The ID to check
   * @returns True if the ID exists, false otherwise
   */
  has(id: number): boolean {
    return this._index.has(id);
  }

  /** Get a string representation of the index
   * @returns String describing the index
   */
  toString(): string {
    return this._index.toString();
  }

  // Property accessors

  /** The Space used to calculate distances */
  get space(): Space {
    return this._index.space;
  }

  /** The number of dimensions in each vector */
  get numDimensions(): number {
    return this._index.numDimensions;
  }

  /** The M parameter (number of connections) */
  get M(): number {
    return this._index.M;
  }

  /** The efConstruction parameter */
  get efConstruction(): number {
    return this._index.efConstruction;
  }

  /** The maximum number of elements that can be stored */
  get maxElements(): number {
    return this._index.maxElements;
  }

  /** Set the maximum number of elements (resizes the index) */
  set maxElements(value: number) {
    this._index.maxElements = value;
  }

  /** The storage data type used by this index */
  get storageDataType(): StorageDataType {
    return this._index.storageDataType;
  }

  /** The current number of elements in the index */
  get numElements(): number {
    return this._index.numElements;
  }

  /** Array of all non-deleted IDs in the index */
  get ids(): number[] {
    return this._index.ids;
  }

  /** The default search depth for queries */
  get ef(): number {
    return this._index.ef;
  }

  /** Set the default search depth for queries */
  set ef(value: number) {
    this._index.ef = value;
  }

  /** The number of non-deleted vectors in the index.
   * This is an alias for the number of IDs in the index.
   */
  get length(): number {
    return this._index.length;
  }
}

// Export the native enums from the C+ module
export const { Space: NativeSpace, StorageDataType: NativeStorageDataType } =
  native;

// For testing purposes (ESM compatible check)
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log("Space enum:", Space);
  console.log("StorageDataType enum:", StorageDataType);
  console.log("Index class available:", typeof Index);
  console.log("\nModule exports are ready for import:");
  console.log('  import { Index, Space, StorageDataType } from "voyager-node"');
}
