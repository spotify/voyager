import { Index, Space, StorageDataType } from "../src/voyager-node.ts";

interface TestConfig {
  numDimensions: number;
  numElements: number;
}

interface SpaceConfig {
  space: Space;
  name: string;
}

interface StorageTypeConfig {
  type: StorageDataType;
  name: string;
}

// Test parameters matching the Python tests
const testConfigs: TestConfig[] = [
  { numDimensions: 4, numElements: 1024 },
  { numDimensions: 16, numElements: 1024 },
  { numDimensions: 128, numElements: 512 },
  { numDimensions: 256, numElements: 256 },
  { numDimensions: 4096, numElements: 128 },
];

const spaces: SpaceConfig[] = [
  { space: Space.Euclidean, name: "Euclidean" },
  { space: Space.Cosine, name: "Cosine" },
];

const storageTypes: StorageTypeConfig[] = [
  { type: StorageDataType.E4M3, name: "E4M3" },
  { type: StorageDataType.Float8, name: "Float8" },
  { type: StorageDataType.Float32, name: "Float32" },
];

function generateRandomData(
  numElements: number,
  numDimensions: number
): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < numElements; i++) {
    const vector: number[] = [];
    for (let j = 0; j < numDimensions; j++) {
      // Generate random values in range [-1, 1]
      vector.push(Math.random() * 2 - 1);
    }
    data.push(vector);
  }
  return data;
}

function arraysAreClose(
  a: number[],
  b: number[],
  atol: number = 1e-2,
  rtol: number = 0.08
): boolean {
  if (a.length !== b.length) return false;

  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    const threshold = atol + rtol * Math.abs(b[i]);
    if (diff > threshold) {
      return false;
    }
  }
  return true;
}

function testRecreateIndex(
  numDimensions: number,
  numElements: number,
  space: Space,
  spaceName: string,
  storageDataType: StorageDataType,
  storageTypeName: string
): boolean {
  const testName = `Index recreation: ${numDimensions}D ${numElements} elements, ${spaceName}, ${storageTypeName}`;

  try {
    // Generate random input data
    const inputData = generateRandomData(numElements, numDimensions);

    // Create original index
    const index = new Index({
      space,
      numDimensions,
      efConstruction: numElements,
      M: 20,
      storageDataType,
      maxElements: numElements,
    });

    // Add items to original index
    const ids = index.addItems(inputData);

    // Verify all IDs are present
    const indexIds = index.ids.sort((a: number, b: number) => a - b);
    const sortedIds = [...ids].sort((a: number, b: number) => a - b);

    if (JSON.stringify(indexIds) !== JSON.stringify(sortedIds)) {
      console.error(`✗ ${testName}`);
      console.error("  ID mismatch after adding items");
      return false;
    }

    // Create recreated index with same parameters
    const recreated = new Index({
      space: index.space,
      numDimensions: index.numDimensions,
      M: index.M,
      efConstruction: index.efConstruction,
      maxElements: index.length,
      storageDataType: index.storageDataType,
    });

    // Get vectors in order and add to recreated index
    const orderedIds = index.ids;
    const vectors = index.getVectors(orderedIds);
    recreated.addItems(vectors, orderedIds);

    // Verify each ID exists in both indices and vectors match
    let mismatchCount = 0;
    for (const id of ids) {
      // Check ID exists in both
      if (!index.has(id)) {
        console.error(`  ID ${id} missing from original index`);
        mismatchCount++;
        continue;
      }
      if (!recreated.has(id)) {
        console.error(`  ID ${id} missing from recreated index`);
        mismatchCount++;
        continue;
      }
      // Compare vectors
      const originalVector = index.getVector(id);
      const recreatedVector = recreated.getVector(id);
      if (!arraysAreClose(originalVector, recreatedVector)) {
        mismatchCount++;
        if (mismatchCount < 3) {
          // Only log first few mismatches
          console.error(`  Vector mismatch for ID ${id}`);
          console.error(
            `    Original: [${originalVector
              .slice(0, 5)
              .map((v: number) => v.toFixed(4))
              .join(", ")}...]`
          );
          console.error(
            `    Recreated: [${recreatedVector
              .slice(0, 5)
              .map((v: number) => v.toFixed(4))
              .join(", ")}...]`
          );
        }
      }
    }

    if (mismatchCount > 0) {
      console.error(`✗ ${testName}`);
      console.error(`  ${mismatchCount} vector mismatches found`);
      return false;
    }

    console.log(`✓ ${testName}`);
    return true;
  } catch (error) {
    console.error(`✗ ${testName}`);
    console.error(`  Error: ${error}`);
    return false;
  }
}

export default function runAllTests(): boolean {
  console.log("Running index recreation tests...\n");
  console.log("Note: These tests may take a while for large dimensions.\n");

  let totalTests = 0;
  let passedTests = 0;

  for (const { numDimensions, numElements } of testConfigs) {
    for (const { space, name: spaceName } of spaces) {
      for (const { type, name: storageTypeName } of storageTypes) {
        totalTests++;
        if (
          testRecreateIndex(
            numDimensions,
            numElements,
            space,
            spaceName,
            type,
            storageTypeName
          )
        ) {
          passedTests++;
        }
      }
    }
  }

  const failed = totalTests - passedTests;
  console.log("\n=== Index Recreation Test Summary ===");
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${failed}`);
  console.log(`Total: ${totalTests}`);

  if (failed === 0) {
    console.log("✓ All index recreation tests passed!");
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
