import { Index, Space, StorageDataType } from "../src/voyager-node.ts";

/**
 * IMPORTANT NOTE:
 * The Node.js bindings do NOT expose the E4M3T class that is available in Python.
 *
 * In Python, the E4M3T class provides low-level access to E4M3 number properties:
 * - EM3T.from_char(byte) - create from raw byte
 * - E4M3(float) - create from float
 * - float(e4m3) - convert to float
 * - Properties: sign, exponent, mantissa, raw_exponent, raw_mantissa, size
 *
 * The Node js bindings focus on the practical Index API instead. Users work with
 * regular JavaScript number arrays, and the Index handles EM3 conversion internally.
 *
 * Therefore, these tests focus on:
 * 1. Creating indices with E4M3 storage type
 * 2. Verifying that vectors are stored and retrieved correctly
 * 3. Testing the quantization behavior (precision loss) of E4M3
 * 4. Testing edge cases like range limits and special values
 *
 * For detailed E4M3 implementation testing (bit-level operations, rounding modes, etc.),
 * refer to the Python tests or C+ unit tests.
 */

// Helper functions
function assertClose(
  actual: number,
  expected: number,
  tolerance: number,
  message?: string
): void {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    throw new Error(
      `${
        message || "Assertion failed"
      }: Expected ${expected}, got ${actual}, diff ${diff} > tolerance ${tolerance}`
    );
  }
}

function assertArrayClose(
  actual: number[],
  expected: number[],
  tolerance: number,
  message?: string
): void {
  if (actual.length !== expected.length) {
    throw new Error(
      `${message || "Assertion failed"}: Array lengths differ: ${
        actual.length
      } vs ${expected.length}`
    );
  }
  for (let i = 0; i < actual.length; i++) {
    assertClose(
      actual[i],
      expected[i],
      tolerance,
      `${message || "Assertion failed"} at index ${i}`
    );
  }
}

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function assertEqual<T>(actual: T, expected: T, message?: string): void {
  if (actual !== expected) {
    throw new Error(
      `${message || "Assertion failed"}: Expected ${expected}, got ${actual}`
    );
  }
}

/**
 * Normalize a vector (for cosine distance)
 */
function normalize(vec: number[]): number[] {
  const sumSquares = vec.reduce((sum, val) => sum + val * val, 0);
  const magnitude = Math.sqrt(sumSquares) + 1e-30;
  return vec.map((val) => val / magnitude);
}

/**
 * Generate a range of numbers similar to Python's np.arange
 */
function* arange(start: number, end: number, step: number): Generator<number> {
  for (let i = start; i < end; i += step) {
    yield i;
  }
}

// Test data: Known E4M3 values from the Python implementation
// E4M3 has limited precision, so we test with values that should roundtrip exactly
const EXACT_E4M3_VALUES = [
  0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128,
  256, -0.015625, -0.03125, -0.0625, -0.125, -0.25, -0.5, -1, -2, -4, -8, -16,
  -32, -64, -128, -256, 0.0009765625, 0.017578125, 0.03515625, 0.0703125,
  0.140625, 0.28125, 0.5625, 1.125, -0.0009765625, -0.017578125, -0.03515625,
  -0.0703125, -0.140625, -0.28125, -0.5625, -1.125,
];

/**
 * Test 1: Basic E4M3 Index Creation
 * Verify that we can create an index with E4M3 storage type
 */
function testE4M3IndexCreation(): void {
  console.log("Testing E4M3 index creation...");
  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 10,
    storageDataType: StorageDataType.E4M3,
    M: 16,
    efConstruction: 200,
    randomSeed: 1,
    maxElements: 100,
  });

  assertEqual(
    index.storageDataType,
    StorageDataType.E4M3,
    "Storage type should be E4M3"
  );
  assertEqual(index.numDimensions, 10, "Number of dimensions should be 10");
  assertEqual(index.numElements, 0, "Index should be empty initially");

  console.log("✓ E4M3 index creation test passed");
}

/**
 * Test 2: Adding and Retrieving Vectors with E4M3
 * Test that vectors can be added and retrieved (with expected quantization)
 */
function testE4M3VectorStorage(): void {
  console.log("Testing E4M3 vector storage...");

  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 5,
    storageDataType: StorageDataType.E4M3,
  });

  // Use values that should have relatively low quantization error
  const vector = [1.0, -2.0, 0.5, -0.25, 4.0];
  const id = index.addItem(vector);
  const retrieved = index.getVector(id);

  // E4M3 has limited precision, so we expect some quantization error
  // Maximum error for values in this range should be around 0.5 or less
  const maxError = 0.5;
  assertArrayClose(
    retrieved,
    vector,
    maxError,
    "Retrieved vector should be close to original"
  );

  console.log("✓ E4M3 vector storage test passed");
}

/**
 * Test 3: E4M3 Range Testing
 * Test vectors within valid E4M3 range [-448, 448]
 */
function testE4M3Range(): void {
  console.log("Testing E4M3 range...");
  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 3,
    storageDataType: StorageDataType.E4M3,
  });

  // Test values across the valid range
  const testRanges = [
    [-10, 10, 1],
    [-100, 100, 10],
    [-448, 448, 50],
  ];

  for (const [min, max, step] of testRanges) {
    const values = Array.from(arange(min, max, step));
    for (const value of values) {
      const vector = [value, value / 2, value / 4];
      const id = index.addItem(vector);
      const retrieved = index.getVector(id);

      // Calculate expected maximum error based on value magnitude
      // E4M3 error increases with magnitude
      const expectedError = Math.max(16, Math.abs(value) * 0.1);

      assertArrayClose(
        retrieved,
        vector,
        expectedError,
        `Vector with values around ${value} should be retrieved within error tolerance`
      );
    }
  }

  console.log("✓ E4M3 range test passed");
}

/**
 * Test 4: E4M3 with Small Values
 * Test precision for small values (near zero)
 */
function testE4M3SmallValues(): void {
  console.log("Testing E4M3 with small values...");
  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 4,
    storageDataType: StorageDataType.E4M3,
  });

  // Test small values where E4M3 should have better relative precision
  const smallValues = [0.01, -0.01, 0.001, -0.002, 0.1, -0.2, 0.5, -0.5];

  for (const value of smallValues) {
    const vector = [value, value * 2, value * 3, value * 4];
    const id = index.addItem(vector);
    const retrieved = index.getVector(id);

    // For small values, error should be much smaller
    const expectedError = 0.05;
    assertArrayClose(
      retrieved,
      vector,
      expectedError,
      `Small value vector ${value} should have low quantization error`
    );
  }

  console.log("✓ E4M3 small values test passed");
}

/**
 * Test 5: E4M3 with Cosine Space
 * Test the real-world vector from Python tests with Cosine space
 */
function testE4M3Cosine(): void {
  console.log("Testing E4M3 with Cosine space...");

  // Real-world vector from Python test
  const REAL_WORLD_VECTOR = [
    -0.28728199005126953, -0.4670010209083557, 0.2676819860935211,
    -0.1626259982585907, -0.6251270174980164, 0.2816449999809265,
    0.32270801067352295, 0.33403000235557556, 0.7520139813423157,
    0.5022000074386597, 0.7720339894294739, -0.5909199714660645,
    0.5918650031089783, -0.15842899680137634, -0.11246500164270401,
    0.24038001894950867, -1.157925009727478, -0.16482099890708923,
    0.09613300859928131, 0.5384849905967712, 0.17511099576950073,
    0.09210799634456635, -0.2158990055322647, -0.1197270005941391,
    0.5386099815368652, 0.196150004863739, -0.8914260864257812,
    -0.19836701452732086, 0.3211739957332611, 0.33692699670791626,
    0.620635986328125, -0.8655009865760803, -0.2893890142440796,
    0.2558070123195648, -0.0019950000569224358, 0.25856301188468933,
    0.831616997718811, 1.3858330249786377, -0.5884850025177002,
    -0.24664302170276642, 0.00035700001171790063, 0.8199999928474426,
    -0.1729460060596466, 0.6167529821395874, 0.1001340001821518,
    0.2342749983072281, 0.47478801012039185, 0.6487500071525574,
    0.3548029959201813, 0.2365729957818985, -0.713392972946167,
    -0.9608209729194641, -0.09217199683189392, -0.0563880018889904,
    -0.022280000150203705, -0.3831019997596741, -0.10219399631023407,
    -0.1772879958152771, -0.2045920193195343, -0.5201849937438965,
    -1.6222929954528809, 0.7166309952735901, -0.3722609877586365,
    -0.4575370252132416, 0.5124289989471436, 0.02841399982571602,
    0.06806100159883499, -0.2725119888782501, -0.5817689895629883,
    -0.2708030045032501, 1.121297001838684, -0.639868974685669,
    0.39189401268959045, -0.1527390033006668, 0.6738319993019104,
    -0.7513130307197571, 0.23471000790596008, -0.8855159878730774,
    0.7264220118522644, 0.4370560348033905,
  ];

  const index = new Index({
    space: Space.Cosine,
    numDimensions: 80,
    storageDataType: StorageDataType.E4M3,
  });

  const id = index.addItem(REAL_WORLD_VECTOR);
  const retrieved = index.getVector(id);

  // When using Cosine space, the vector is normalized before storage
  // So we compare against the normalized version
  const normalized_vector = normalize(REAL_WORLD_VECTOR);

  // E4M3 quantization will cause some error
  // We expect higher error due to the normalization + quantization
  const maxError = 0.1;

  assertArrayClose(
    retrieved,
    normalized_vector,
    maxError,
    "Retrieved vector should be close to normalized original with E4M3 quantization"
  );

  console.log("✓ E4M3 Cosine space test passed");
}

/**
 * Test 6: E4M3 Query Accuracy
 * Test that queries work correctly with E4M3 storage
 */
function testE4M3QueryAccuracy(): void {
  console.log("Testing E4M3 query accuracy...");

  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 16,
    storageDataType: StorageDataType.E4M3,
    M: 16,
    efConstruction: 200,
  });

  // Add several vectors
  const vectors: number[][] = [];
  const numVectors = 100;
  for (let i = 0; i < numVectors; i++) {
    const vector: number[] = Array.from(
      { length: 16 },
      () => Math.random() * 20 - 10 // Range [-10, 10]
    );
    vectors.push(vector);
    index.addItem(vector, i);
  }

  // Query with one of the vectors
  const queryVector = vectors[0];
  const results = index.query(queryVector, 5);

  // The first result should be the query vector itself (or very close)
  assertEqual(
    results.neighbors[0],
    0,
    "First neighbor should be the query vector itself"
  );
  assert(results.distances[0] < 1.0, "Distance to self should be very small");

  console.log("✓ E4M3 query accuracy test passed");
}

/**
 * Test 7: E4M3 Monotonic Behavior
 * Test that similar values map to the same or increasing E4M3 values
 */
function testE4M3MonotonicBehavior(): void {
  console.log("Testing E4M3 monotonic behavior...");

  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 1,
    storageDataType: StorageDataType.E4M3,
  });

  // Test that increasing values result in increasing or equal stored values
  const testValues = Array.from(arange(-100, 100, 0.5));
  let prevRetrieved = -Infinity;

  for (const value of testValues) {
    const id = index.addItem([value]);
    const retrieved = index.getVector(id)[0];

    assert(
      retrieved >= prevRetrieved - 1e-6, // Allow tiny floating point errors
      `E4M3 should maintain monotonic order: ${prevRetrieved} should be <= ${retrieved} for input ${value}`
    );
    prevRetrieved = retrieved;
  }

  console.log("✓ E4M3 monotonic behavior test passed");
}

/**
 * Test 8: E4M3 Batch Operations
 * Test adding multiple vectors at once with E4M3
 */
function testE4M3BatchOperations(): void {
  console.log("Testing E4M3 batch operations...");

  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 8,
    storageDataType: StorageDataType.E4M3,
  });

  // Create batch of vectors
  const batchSize = 50;
  const vectors: number[][] = [];

  for (let i = 0; i < batchSize; i++) {
    const vector: number[] = [];
    for (let j = 0; j < 8; j++) {
      vector.push(Math.random() * 20 - 10);
    }
    vectors.push(vector);
  }

  // Add vectors in batch
  const ids = index.addItems(vectors);

  assertEqual(ids.length, batchSize, "Should return correct number of IDs");
  assertEqual(index.numElements, batchSize, "Index should contain all vectors");

  // Verify all vectors can be retrieved
  const retrieved = index.getVectors(ids);
  assertEqual(retrieved.length, batchSize, "Should retrieve all vectors");

  // Check that each vector is approximately correct
  for (let i = 0; i < batchSize; i++) {
    assertArrayClose(
      retrieved[i],
      vectors[i],
      2.0, // Allow reasonable error for E4M3
      `Batch vector ${i} should be retrievable`
    );
  }

  console.log("✓ E4M3 batch operations test passed");
}

/**
 * Test 9: E4M3 Edge Cases
 * Test edge cases like zero, near-zero, and boundary values
 */
function testE4M3EdgeCases(): void {
  console.log("Testing E4M3 edge cases...");

  const index = new Index({
    space: Space.Euclidean,
    numDimensions: 3,
    storageDataType: StorageDataType.E4M3,
  });

  // Test zero vector
  const zeroVector = [0, 0, 0];
  const id1 = index.addItem(zeroVector);
  const retrieved1 = index.getVector(id1);
  assertArrayClose(retrieved1, zeroVector, 1e-6, "Zero vector should be exact");

  // Test mixed positive/negative
  const mixedVector = [1.0, -1.0, 0.5];
  const id2 = index.addItem(mixedVector);
  const retrieved2 = index.getVector(id2);
  assertArrayClose(
    retrieved2,
    mixedVector,
    0.1,
    "Mixed sign vector should be close"
  );

  // Test near-boundary values (E4M3 range is [-448, 448])
  const nearBoundary = [400, -400, 200];
  const id3 = index.addItem(nearBoundary);
  const retrieved3 = index.getVector(id3);
  assertArrayClose(
    retrieved3,
    nearBoundary,
    50,
    "Near-boundary vector should be within tolerance"
  );

  console.log("✓ E4M3 edge cases test passed");
}

/**
 * Test 10: E4M3 Storage vs Float32 Comparison
 * Compare behavior between E4M3 and Float32 storage
 */
function testE4M3vsFloat32(): void {
  console.log("Testing E4M3 vs Float32 comparison...");

  const indexE4M3 = new Index({
    space: Space.Euclidean,
    numDimensions: 10,
    storageDataType: StorageDataType.E4M3,
  });

  const indexFloat32 = new Index({
    space: Space.Euclidean,
    numDimensions: 10,
    storageDataType: StorageDataType.Float32,
  });

  // Same vector to both indices
  const vector = [1.5, -2.3, 0.7, -0.4, 3.2, -1.1, 0.9, -0.6, 2.8, -3.5];

  indexE4M3.addItem(vector, 0);
  indexFloat32.addItem(vector, 0);

  const retrievedE4M3 = indexE4M3.getVector(0);
  const retrievedFloat32 = indexFloat32.getVector(0);

  // Float32 should be more precise
  assertArrayClose(
    retrievedFloat32,
    vector,
    1e-6,
    "Float32 should be very precise"
  );

  // E4M3 should have more error but still be reasonable
  assertArrayClose(
    retrievedE4M3,
    vector,
    0.5,
    "E4M3 should be within tolerance"
  );

  // E4M3 and Float32 should differ due to quantization
  let hasDifference = false;
  for (let i = 0; i < vector.length; i++) {
    if (Math.abs(retrievedE4M3[i] - retrievedFloat32[i]) > 1e-6) {
      hasDifference = true;
      break;
    }
  }
  assert(
    hasDifference,
    "E4M3 and Float32 storage should produce different results due to quantization"
  );

  console.log("✓ E4M3 vs Float32 comparison test passed");
}

// Main test runner
export default function runAllTests(): boolean {
  console.log("\n=== Running E4M3 Tests ===\n");

  const tests = [
    testE4M3IndexCreation,
    testE4M3VectorStorage,
    testE4M3Range,
    testE4M3SmallValues,
    testE4M3Cosine,
    testE4M3QueryAccuracy,
    testE4M3MonotonicBehavior,
    testE4M3BatchOperations,
    testE4M3EdgeCases,
    testE4M3vsFloat32,
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    try {
      test();
      passed++;
    } catch (error) {
      failed++;
      console.error(`Test ${test.name} failed:`, error);
    }
  }

  console.log("\n=== E4M3 Test Summary ===");
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  console.log(`Total:  ${tests.length}`);

  if (failed === 0) {
    console.log("✓ All E4M3 tests passed!");
    return true;
  } else {
    console.log(`✗ ${failed} tests failed`);
    throw new Error(`${failed} tests failed`);
  }
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    runAllTests();
  } catch (error) {
    process.exit(1);
  }
}
