import { Index, Space, StorageDataType } from "../src/voyager-node.ts";
import fs from "fs";
import path from "path";
import os from "os";

// Seeded random number generator for deterministic tests
class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  // Simple LCG (Linear Congruential Generator)
  next(): number {
    this.seed = (this.seed * 1664525 + 1013904223) % 4294967296;
    return this.seed / 4294967296;
  }

  // Generate random number in range (min, max)
  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }
}

// Helper functions
function generateRandomData(
  numElements: number,
  numDimensions: number,
  rng?: SeededRandom
): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < numElements; i++) {
    const vector: number[] = [];
    for (let j = 0; j < numDimensions; j++) {
      const randomValue = rng ? rng.range(-1, 1) : Math.random() * 2 - 1;
      vector.push(randomValue); // Range [-1, 1]
    }
    data.push(vector);
  }
  return data;
}

function roundToFloat8(data: number[][]): number[][] {
  return data.map((vector) => vector.map((v) => Math.round(v * 127) / 127));
}

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

function createTempFile(suffix: string = ".voy"): string {
  const tmpDir = os.tmpdir();
  const fileName = `voyager_test_${Date.now()}_${Math.random()
    .toString(36)
    .substring(7)}${suffix}`;
  return path.join(tmpDir, fileName);
}

function assertSetsEqual<T>(
  actual: Set<T>,
  expected: Set<T>,
  message?: string
): void {
  if (actual.size !== expected.size) {
    throw new Error(
      `${message || "Sets not equal"}: sizes differ ${actual.size} vs ${
        expected.size
      }`
    );
  }
  for (const item of expected) {
    if (!actual.has(item)) {
      throw new Error(
        `${
          message || "Sets not equal"
        }: expected item ${item} not found in actual set`
      );
    }
  }
}

// Test parameters
interface TestParams {
  numDimensions: number;
  numElements: number;
  space: Space;
  spaceName: string;
  storageDataType: StorageDataType;
  storageTypeName: string;
  distanceTolerance: number;
  recallTolerance: number;
}

const testConfigurations: TestParams[] = [];

// Generate test configurations
const dimensions = [4, 16, 128, 256, 4096];
const elements = [1024, 1024, 512, 256, 128];
const spaces = [
  { space: Space.Euclidean, name: "Euclidean" },
  { space: Space.Cosine, name: "Cosine" },
];
const storageTypes = [
  { type: StorageDataType.E4M3, distTol: 0.03, recallTol: 0.4, name: "E4M3" },
  {
    type: StorageDataType.Float8,
    distTol: 0.03,
    recallTol: 0.5,
    name: "Float8",
  },
  {
    type: StorageDataType.Float32,
    distTol: 2e-7,
    recallTol: 1.0,
    name: "Float32",
  },
];

for (let i = 0; i < dimensions.length; i++) {
  for (const space of spaces) {
    for (const storage of storageTypes) {
      testConfigurations.push({
        numDimensions: dimensions[i],
        numElements: elements[i],
        space: space.space,
        spaceName: space.name,
        storageDataType: storage.type,
        storageTypeName: storage.name,
        distanceTolerance: storage.distTol,
        recallTolerance: storage.recallTol,
      });
    }
  }
}

function testCreateAndQuery(params: TestParams): boolean {
  const testName = `Create and query: ${params.numDimensions}D, ${params.numElements} elements, ${params.spaceName}, ${params.storageTypeName}`;

  try {
    let inputData = generateRandomData(
      params.numElements,
      params.numDimensions
    );

    // Apply Float8 quantization if needed
    if (params.storageDataType === StorageDataType.Float8) {
      inputData = roundToFloat8(inputData);
    }

    const ids = Array.from({ length: inputData.length }, (_, i) => i);

    const index = new Index({
      space: params.space,
      numDimensions: params.numDimensions,
      efConstruction: params.numElements,
      M: 20,
      storageDataType: params.storageDataType,
    });

    // Check repr contains expected
    const indexStr = index.toString();
    assert(
      indexStr.includes(params.spaceName),
      "Index string should contain space name"
    );
    assert(
      indexStr.includes(params.numDimensions.toString()),
      "Index string should contain dimensions"
    );
    assert(
      indexStr.includes(params.storageTypeName),
      "Index string should contain storage type"
    );

    index.ef = params.numElements;

    const addedIds = index.addItems(inputData, ids);
    assertEqual(
      addedIds.length,
      ids.length,
      "Added IDs length should match input IDs"
    );
    for (let i = 0; i < ids.length; i++) {
      assertEqual(addedIds[i], ids[i], `Added ID at index ${i} should match`);
    }

    assertEqual(
      index.length,
      params.numElements,
      "Index length should match numElements"
    );
    assertEqual(
      index.numElements,
      params.numElements,
      "Index numElements should match"
    );

    // Query all vectors and check recall
    const result = index.query(inputData, 1);
    const labels = result.neighbors;
    const distances = result.distances;

    let matches = 0;
    for (let i = 0; i < labels.length; i++) {
      if (labels[i][0] === i) matches++;
    }

    const recall = matches / inputData.length;
    assert(
      recall >= params.recallTolerance,
      `Recall ${recall} should be ≥ ${params.recallTolerance}`
    );

    // Check distances are close to zero
    const tolerance = params.distanceTolerance * params.numDimensions;
    for (let i = 0; i < distances.length; i++) {
      assertClose(distances[i][0], 0, tolerance, `Distance for vector ${i}`);
    }

    // Check IDs set
    const indexIds = new Set(index.ids);
    const expectedIds = new Set(ids);
    assertSetsEqual(indexIds, expectedIds, "Index IDs should match input IDs");

    // Test single-query interface
    for (let i = 0; i < Math.min(10, inputData.length); i++) {
      const singleResult = index.query(inputData[i], 1);
      if (params.storageDataType === StorageDataType.Float32) {
        assertEqual(
          singleResult.neighbors[0],
          i,
          `Single query result for vector ${i}`
        );
        assert(
          singleResult.distances[0] < tolerance,
          `Single query distance for vector ${i} should be < ${tolerance}`
        );
      }
    }

    // Test query with k = all ids
    const allResult = index.query(inputData[0], ids.length);
    const returnedIds = new Set(allResult.neighbors);
    assertSetsEqual(
      returnedIds,
      expectedIds,
      "Query with k=all should return all IDs"
    );

    // Test save and load
    const outputFile = createTempFile();
    try {
      index.saveIndex(outputFile);
      assert(fs.existsSync(outputFile), "Output file should exist");
      const fileSize = fs.statSync(outputFile).size;
      assert(fileSize > 0, "Output file should not be empty");

      const indexBytes = index.toBuffer();
      assert(indexBytes.length > 0, "Index bytes should not be empty");
      assertEqual(
        indexBytes.length,
        fileSize,
        "Buffer length should match file size"
      );

      const fileBytes = fs.readFileSync(outputFile);
      assertEqual(
        indexBytes.length,
        fileBytes.length,
        "Buffer and file should have same length"
      );
      assert(
        indexBytes.equals(fileBytes),
        "Buffer content should match file content"
      );
    } finally {
      // Cleanup
      if (fs.existsSync(outputFile)) {
        fs.unlinkSync(outputFile);
      }
    }

    console.log(`✓ ${testName}`);
    return true;
  } catch (error) {
    console.error(`✗ ${testName}`);
    console.error(
      `  Error: ${error instanceof Error ? error.message : String(error)}`
    );
    return false;
  }
}

// Test different spaces
function testSpaces(): boolean {
  const testCases = [
    {
      space: Space.Euclidean,
      spaceName: "Euclidean",
      expectedDistances: [0.0, 1.0, 2.0, 2.0, 2.0],
    },
    {
      space: Space.InnerProduct,
      spaceName: "InnerProduct",
      expectedDistances: [-2.0, -1.0, 0.0, 0.0, 0.0],
    },
    {
      space: Space.Cosine,
      spaceName: "Cosine",
      expectedDistances: [0, 1.835e-1, 4.23e-1, 4.23e-1, 4.23e-1],
    },
  ];

  let allPassed = true;

  for (const testCase of testCases) {
    for (let leftDim = 1; leftDim < 32; leftDim += 5) {
      for (let rightDim = 1; rightDim < 128; rightDim += 3) {
        const testName = `Spaces test: ${testCase.spaceName}, leftDim=${leftDim}, rightDim=${rightDim}`;

        try {
          const inputData = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
          ];

          // Pad with zeros
          const data2 = inputData.map((row) => [
            ...Array(leftDim).fill(0),
            ...row,
            ...Array(rightDim).fill(0),
          ]);

          const numDimensions = data2[0].length;
          const index = new Index({
            space: testCase.space,
            numDimensions,
            efConstruction: 100,
            M: 16,
          });
          index.ef = 10;
          index.addItems(data2);

          const result = index.query([data2[data2.length - 1]], 5);
          const distances = result.distances[0];

          assertArrayClose(
            distances,
            testCase.expectedDistances,
            1e-3,
            testName
          );

          console.log(`✓ ${testName}`);
        } catch (error) {
          console.error(`✗ ${testName}`);
          console.error(
            `Error: ${error instanceof Error ? error.message : String(error)}`
          );
          allPassed = false;
        }
      }
    }
  }
  return allPassed;
}

// Test get_vectors
function testGetVectors(): boolean {
  const dimensions = [4, 16, 128, 256, 4096];
  const elements = [1024, 1024, 512, 256, 128];
  const spaces = [
    { space: Space.Euclidean, name: "Euclidean" },
    { space: Space.InnerProduct, name: "InnerProduct" },
  ];

  let allPassed = true;
  for (let i = 0; i < dimensions.length; i++) {
    for (const space of spaces) {
      const testName = `GetVectors: ${dimensions[i]}D, ${elements[i]} elements, ${space.name}`;

      try {
        const inputData = generateRandomData(elements[i], dimensions[i]);
        const index = new Index({
          space: space.space,
          numDimensions: dimensions[i],
        });

        const labels = Array.from({ length: elements[i] }, (_, idx) => idx);

        // Before adding anything, getting any labels should fail
        try {
          index.getVectors(labels);
          throw new Error(
            "Should have thrown an error when getting vectors before adding"
          );
        } catch (error) {
          // Expected to throw
        }

        index.addItems(inputData, labels);

        // Test single vector retrieval
        for (let j = 0; j < Math.min(10, labels.length); j++) {
          const expectedVector = inputData[labels[j]];
          const actualVector = index.getVector(labels[j]);
          assertArrayClose(
            actualVector,
            expectedVector,
            1e-6,
            `Vector ${labels[j]}`
          );
        }

        // Test batch vector retrieval
        const vectors = index.getVectors(labels);
        assertEqual(
          vectors.length,
          inputData.length,
          "Retrieved vectors length"
        );

        for (let j = 0; j < vectors.length; j++) {
          assertArrayClose(vectors[j], inputData[j], 1e-6, `Batch vector ${j}`);
        }

        console.log(`✓ ${testName}`);
      } catch (error) {
        console.error(`✗ ${testName}`);
        console.error(
          `Error: ${error instanceof Error ? error.message : String(error)}`
        );
        allPassed = false;
      }
    }
  }
  return allPassed;
}

// Test load from buffer
function testLoadFromBuffer(): boolean {
  const dimensions = [4, 16, 128, 256, 4096];
  const elements = [1024, 1024, 512, 256, 128];
  const spaces = [
    { space: Space.Euclidean, name: "Euclidean" },
    { space: Space.Cosine, name: "Cosine" },
  ];
  const storageTypes = [
    { type: StorageDataType.Float8, name: "Float8" },
    { type: StorageDataType.Float32, name: "Float32" },
  ];

  let allPassed = true;

  for (let i = 0; i < dimensions.length; i++) {
    for (const space of spaces) {
      for (const storage of storageTypes) {
        const testName = `LoadFromBuffer: ${dimensions[i]}D, ${elements[i]} elements, ${space.name}, ${storage.name}`;

        try {
          let inputData = generateRandomData(elements[i], dimensions[i]);

          if (storage.type === StorageDataType.Float8) {
            inputData = roundToFloat8(inputData);
          }

          const index = new Index({
            space: space.space,
            numDimensions: dimensions[i],
            efConstruction: elements[i],
            M: 20,
            storageDataType: storage.type,
          });

          index.addItems(inputData);

          const buffer = index.toBuffer();
          const reloaded = Index.fromBuffer(buffer, {
            space: space.space,
            numDimensions: dimensions[i],
            storageDataType: storage.type,
          });

          const labels = Array.from({ length: elements[i] }, (_, idx) => idx);
          const originalVectors = index.getVectors(labels);
          const reloadedVectors = reloaded.getVectors(labels);

          assertEqual(
            originalVectors.length,
            reloadedVectors.length,
            "Vector count"
          );

          for (let j = 0; j < originalVectors.length; j++) {
            assertArrayClose(
              originalVectors[j],
              reloadedVectors[j],
              1e-6,
              `Vector ${j}`
            );
          }

          console.log(`✓ ${testName}`);
        } catch (error) {
          console.error(`✗ ${testName}`);
          console.error(
            `Error: ${error instanceof Error ? error.message : String(error)}`
          );
          allPassed = false;
        }
      }
    }
  }
  return allPassed;
}

// Test query_ef parameter
function testQueryEf(): boolean {
  const spaces = [
    { space: Space.Euclidean, name: "Euclidean" },
    { space: Space.Cosine, name: "Cosine" },
  ];
  const queryEfParams = [
    { queryEf: 1, rankTolerance: 500 },
    { queryEf: 2, rankTolerance: 75 },
    { queryEf: 100, rankTolerance: 1 },
  ];

  let allPassed = true;

  for (const space of spaces) {
    for (const params of queryEfParams) {
      const testName = `QueryEf: ${space.name}, queryEf=${params.queryEf}, rankTolerance=${params.rankTolerance}`;

      try {
        const numDimensions = 32;
        const numElements = 1000;
        const inputData = generateRandomData(numElements, numDimensions);

        const index = new Index({
          space: space.space,
          numDimensions,
          efConstruction: numElements,
          M: 20,
          randomSeed: 123,
        });

        index.ef = numElements;
        index.addItems(inputData);

        // Query with high query_ef to get "correct" results
        const correctResults = index.query(
          inputData,
          numElements,
          -1,
          numElements
        );
        const closestLabelsPerVector = correctResults.neighbors;

        // Query with specified query_ef
        const testResults = index.query(inputData, 1, -1, params.queryEf);
        const labels = testResults.neighbors;

        for (let vectorIndex = 0; vectorIndex < labels.length; vectorIndex++) {
          const returnedLabel = labels[vectorIndex][0];
          const actualRank =
            closestLabelsPerVector[vectorIndex].indexOf(returnedLabel);
          assert(
            actualRank < params.rankTolerance,
            `Vector ${vectorIndex}: rank ${actualRank} should be < ${params.rankTolerance}`
          );
        }

        // Test single-query interface
        for (
          let vectorIndex = 0;
          vectorIndex < Math.min(10, inputData.length);
          vectorIndex++
        ) {
          const singleResult = index.query(
            inputData[vectorIndex],
            1,
            -1,
            params.queryEf
          );
          const returnedLabel = singleResult.neighbors[0];
          const actualRank =
            closestLabelsPerVector[vectorIndex].indexOf(returnedLabel);
          assert(
            actualRank < params.rankTolerance,
            `Single query vector ${vectorIndex}: rank ${actualRank} should be < ${params.rankTolerance}`
          );
        }

        console.log(`✓ ${testName}`);
      } catch (error) {
        console.error(`✗ ${testName}`);
        console.error(
          `Error: ${error instanceof Error ? error.message : String(error)}`
        );
        allPassed = false;
      }
    }
  }
  return allPassed;
}

// Test add_single_item
function testAddSingleItem(): boolean {
  const numDimensions = 4;
  const elementCounts = [100, 1000];
  const spaces = [
    { space: Space.Euclidean, name: "Euclidean" },
    { space: Space.InnerProduct, name: "InnerProduct" },
  ];
  const storageTypes = [
    { type: StorageDataType.Float8, tolerance: 0.01, name: "Float8" },
    { type: StorageDataType.Float32, tolerance: 0.01, name: "Float32" },
    { type: StorageDataType.E4M3, tolerance: 0.075, name: "E4M3" },
  ];

  let allPassed = true;

  for (const numElements of elementCounts) {
    for (const space of spaces) {
      for (const storage of storageTypes) {
        const testName = `AddSingleItem: ${numDimensions}D, ${numElements} elements, ${space.name}, ${storage.name}`;

        try {
          const inputData = generateRandomData(numElements, numDimensions);
          const index = new Index({
            space: space.space,
            numDimensions,
            storageDataType: storage.type,
            randomSeed: 123,
          });

          const labels = Array.from({ length: numElements }, (_, i) => i);

          // Before adding anything, getting any labels should fail
          try {
            index.getVectors(labels);
            throw new Error(
              "Should have thrown an error when getting vectors before adding"
            );
          } catch (error) {
            // Expected to throw
          }

          // Add items one by one
          for (let i = 0; i < labels.length; i++) {
            const returnedId = index.addItem(inputData[i], labels[i]);
            assertEqual(returnedId, labels[i], `Returned ID for item ${i}`);
          }

          // Verify vectors
          for (let i = 0; i < labels.length; i++) {
            const expectedVector = inputData[i];
            const actualVector = index.getVector(labels[i]);
            assertArrayClose(
              actualVector,
              expectedVector,
              storage.tolerance,
              `Vector ${labels[i]}`
            );
          }

          // Test batch retrieval
          const vectors = index.getVectors(labels);
          for (let i = 0; i < vectors.length; i++) {
            assertArrayClose(
              vectors[i],
              inputData[i],
              storage.tolerance,
              `Batch vector ${i}`
            );
          }

          console.log(`✓ ${testName}`);
        } catch (error) {
          console.error(`✗ ${testName}`);
          console.error(
            `Error: ${error instanceof Error ? error.message : String(error)}`
          );
          allPassed = false;
        }
      }
    }
  }

  return allPassed;
}

// Test accuracy for inner product
function testAccuracyForInnerProduct(): boolean {
  const testName = "AccuracyForInnerProduct";
  try {
    const space = Space.InnerProduct;
    const numDimensions = 1024;
    const numElements = 10000;

    // Use seeded random number generator for deterministic results
    const rng = new SeededRandom(1);

    // Generate random data with seeded RNG (values in [0, 1) like Python)
    const inputData: number[][] = [];
    for (let i = 0; i < numElements; i++) {
      const vector: number[] = [];
      for (let j = 0; j < numDimensions; j++) {
        vector.push(rng.next()); // Range [0, 1)
      }
      inputData.push(vector);
    }

    // Create index WITHOUT randomSeed to match Python test
    const index = new Index({
      space,
      numDimensions,
    });

    // Sort the data descending by norm
    const norms = inputData.map((vec) =>
      Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0))
    );
    const indices = Array.from({ length: numElements }, (_, i) => i);
    indices.sort((a, b) => norms[b] - norms[a]);
    const sortedData = indices.map((i) => inputData[i]);

    // Verify sorting
    const sortedNorms = sortedData.map((vec) =>
      Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0))
    );
    for (let i = 1; i < sortedNorms.length; i++) {
      assert(
        sortedNorms[i] <= sortedNorms[i - 1],
        "Norms should be sorted descending"
      );
    }

    // Add items one by one for deterministic ordering
    for (const vector of sortedData) {
      index.addItem(vector);
    }

    // Generate query with seeded RNG
    const query = Array.from({ length: numDimensions }, () => rng.next());

    // Calculate expected distances (inner products)
    const expectedDistances = sortedData.map((vec) =>
      vec.reduce((sum, val, i) => sum + val * query[i], 0)
    );
    const maxDistance = Math.max(...expectedDistances);
    const maxIndex = expectedDistances.indexOf(maxDistance);

    // Query the index
    const result = index.query(query, 3, -1, 10);
    const neighbors = result.neighbors;
    const distances = result.distances;

    assertEqual(
      neighbors[0],
      maxIndex,
      "First neighbor should be the one with max distance"
    );
    assertClose(
      distances[0],
      1.0 - maxDistance,
      0.01,
      "Distance should match expected"
    );

    console.log(`✓ ${testName}`);
    return true;
  } catch (error) {
    console.error(`✗ ${testName}`);
    console.error(
      `Error: ${error instanceof Error ? error.message : String(error)}`
    );
    return false;
  }
}

// Main test runner
export default function runAllTests(): boolean {
  console.log("Running Index Creation Tests...");
  console.log("=".repeat(70));

  let totalTests = 0;
  let passedTests = 0;

  // Run create and query tests
  console.log("\nCreate and Query Tests:");
  console.log("-".repeat(70));
  for (const config of testConfigurations) {
    totalTests++;
    if (testCreateAndQuery(config)) {
      passedTests++;
    }
  }

  // Run spaces tests
  console.log("\nSpaces Tests:");
  console.log("-".repeat(70));
  totalTests++;
  if (testSpaces()) {
    passedTests++;
  }

  // Run get vectors tests
  console.log("\nGet Vectors Tests:");
  console.log("-".repeat(70));
  totalTests++;
  if (testGetVectors()) {
    passedTests++;
  }

  // Run load from buffer tests
  console.log("\nLoad From Buffer Tests:");
  console.log("-".repeat(70));
  totalTests++;
  if (testLoadFromBuffer()) {
    passedTests++;
  }

  // Run query_ef tests
  console.log("\nQuery EF Tests:");
  console.log("-".repeat(70));
  totalTests++;
  if (testQueryEf()) {
    passedTests++;
  }

  // Run add single item tests
  console.log("\nAdd Single Item Tests:");
  console.log("-".repeat(70));
  totalTests++;
  if (testAddSingleItem()) {
    passedTests++;
  }

  // Run accuracy for inner product test
  console.log("\nAccuracy for Inner Product Test:");
  console.log("-".repeat(70));
  totalTests++;
  if (testAccuracyForInnerProduct()) {
    passedTests++;
  }

  const failed = totalTests - passedTests;

  console.log("\n=== Index Creation Test Summary ===");
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${failed}`);
  console.log(`Total: ${totalTests}`);

  if (failed === 0) {
    console.log("✓ All index creation tests passed!");
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
