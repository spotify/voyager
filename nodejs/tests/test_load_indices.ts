import { Index, Space, StorageDataType } from "../src/voyager-node.ts";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// Get the directory name in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const INDEX_FIXTURE_DIR = path.join(
  __dirname,
  "..",
  "..",
  "python",
  "tests",
  "indices"
);

// Helper functions
function detectSpaceFromFilename(filename: string): Space {
  const basename = path.basename(filename).toLowerCase();
  if (basename.includes("cosine")) {
    return Space.Cosine;
  } else if (basename.includes("innerproduct")) {
    return Space.InnerProduct;
  } else if (basename.includes("euclidean")) {
    return Space.Euclidean;
  } else {
    throw new Error(`Not sure which space type is used in ${filename}`);
  }
}

function detectNumDimensionsFromFilename(filename: string): number {
  const basename = path.basename(filename);
  const match = basename.match(/(\d+)dim/);
  if (!match) {
    throw new Error(`Could not detect number of dimensions from ${filename}`);
  }
  return parseInt(match[1], 10);
}

function detectStorageDataTypeFromFilename(filename: string): StorageDataType {
  const basename = path.basename(filename).toLowerCase();
  if (basename.includes("float32")) {
    return StorageDataType.Float32;
  } else if (basename.includes("float8")) {
    return StorageDataType.Float8;
  } else if (basename.includes("e4m3")) {
    return StorageDataType.E4M3;
  } else {
    throw new Error(`Not sure which storage data type is used in ${filename}`);
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
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > tolerance) {
      throw new Error(
        `${message || "Assertion failed"} at index ${i}: Expected ${
          expected[i]
        }, got ${actual[i]}, diff ${diff} > tolerance ${tolerance}`
      );
    }
  }
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

function normalize(vector: number[]): number[] {
  const sumOfSquares = vector.reduce((sum, val) => sum + val * val, 0);
  const magnitude = Math.sqrt(sumOfSquares);
  return vector.map((v) => v / magnitude);
}

// Get all index files from v0 and v1 directories
function getIndexFiles(version: "v0" | "v1"): string[] {
  const versionDir = path.join(INDEX_FIXTURE_DIR, version);
  if (!fs.existsSync(versionDir)) {
    console.warn(`Warning: Directory ${versionDir} does not exist`);
    return [];
  }
  return fs
    .readdirSync(versionDir)
    .filter((f: string) => f.endsWith(".hnsw"))
    .map((f: string) => path.join(versionDir, f));
}

// Test loading V0 indices with parameters
function testLoadV0Index(
  indexFilename: string,
  loadFromBuffer: boolean
): boolean {
  const testName = `Load V0 index ${
    loadFromBuffer ? "from buffer" : "from file"
  }: ${path.basename(indexFilename)}`;

  try {
    const space = detectSpaceFromFilename(indexFilename);
    const numDimensions = detectNumDimensionsFromFilename(indexFilename);
    const storageDataType = detectStorageDataTypeFromFilename(indexFilename);

    let index: Index;
    if (loadFromBuffer) {
      const buffer = fs.readFileSync(indexFilename);
      index = Index.fromBuffer(buffer, {
        space,
        numDimensions,
        storageDataType,
      });
    } else {
      index = Index.loadIndex(indexFilename, {
        space,
        numDimensions,
        storageDataType,
      });
    }

    // All of these test indices are expected to contain exactly 0.0, 0.1, 0.2, 0.3, 0.4
    const expectedIds = new Set([0, 1, 2, 3, 4]);
    const actualIds = new Set(index.ids);
    assertSetsEqual(
      actualIds,
      expectedIds,
      `Index ${path.basename(indexFilename)} has incorrect IDs`
    );

    // Verify vectors
    for (const id of index.ids) {
      const expectedVector = new Array(numDimensions).fill(id * 0.1);
      let expected = expectedVector;

      if (space === Space.Cosine && id > 0) {
        // Voyager stores only normalized vectors in Cosine mode
        expected = normalize(expectedVector);
      }

      const actual = index.getVector(id);
      assertArrayClose(
        actual,
        expected,
        0.2,
        `Vector mismatch for ID ${id} in ${path.basename(indexFilename)}`
      );
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

// Test loading V1 indices (with metadata, no parameters needed)
function testLoadV1Index(
  indexFilename: string,
  loadFromBuffer: boolean
): boolean {
  const testName = `Load V1 index ${
    loadFromBuffer ? "from buffer" : "from file"
  }: ${path.basename(indexFilename)}`;

  try {
    let index: Index;
    if (loadFromBuffer) {
      const buffer = fs.readFileSync(indexFilename);
      index = Index.fromBuffer(buffer);
    } else {
      index = Index.loadIndex(indexFilename);
    }

    const space = index.space;
    const numDimensions = index.numDimensions;

    // All of these test indices are expected to contain exactly 0.0, 0.1, 0.2, 0.3, 0.4
    const expectedIds = new Set([0, 1, 2, 3, 4]);
    const actualIds = new Set(index.ids);
    assertSetsEqual(
      actualIds,
      expectedIds,
      `Index ${path.basename(indexFilename)} has incorrect IDs`
    );

    // Verify vectors
    for (const id of index.ids) {
      const expectedVector = new Array(numDimensions).fill(id * 0.1);
      let expected = expectedVector;

      if (space === Space.Cosine && id > 0) {
        // Voyager stores only normalized vectors in Cosine mode
        expected = normalize(expectedVector);
      }

      const actual = index.getVector(id);
      assertArrayClose(
        actual,
        expected,
        0.2,
        `Vector mismatch for ID ${id} in ${path.basename(indexFilename)}`
      );
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

// Test that V1 indices must have matching parameters if provided
function testV1IndicesMustMatchParameters(
  indexFilename: string,
  loadFromBuffer: boolean
): boolean {
  const validationTestName = `V1 index parameter validation ${
    loadFromBuffer ? "from buffer" : "from file"
  }: ${path.basename(indexFilename)}`;

  try {
    const space = detectSpaceFromFilename(indexFilename);
    const numDimensions = detectNumDimensionsFromFilename(indexFilename);
    const storageDataType = detectStorageDataTypeFromFilename(indexFilename);

    let threwError = false;
    let errorMessage = "";

    try {
      if (loadFromBuffer) {
        const buffer = fs.readFileSync(indexFilename);
        Index.fromBuffer(buffer, {
          space,
          numDimensions: numDimensions + 1, // Intentionally wrong
          storageDataType,
        });
      } else {
        Index.loadIndex(indexFilename, {
          space,
          numDimensions: numDimensions + 1, // Intentionally wrong
          storageDataType,
        });
      }
    } catch (error) {
      threwError = true;
      errorMessage = error instanceof Error ? error.message : String(error);
    }

    if (!threwError) {
      throw new Error(
        "Expected an error to be thrown for mismatched parameters"
      );
    }

    // check that the error message contains information about the dimension mismatch
    if (
      !errorMessage.includes("dimension") ||
      !errorMessage.includes(String(numDimensions)) ||
      !errorMessage.includes(String(numDimensions + 1))
    ) {
      throw new Error(
        `Error message should contain dimension information. Got: ${errorMessage}`
      );
    }

    console.log(`✓ ${validationTestName}`);
    return true;
  } catch (error) {
    console.error(`✗ ${validationTestName}`);
    console.error(
      `  Error: ${error instanceof Error ? error.message : String(error)}`
    );
    return false;
  }
}

// Test loading invalid data (should not crash)
function testLoadingInvalidDataCannotCrash(): boolean {
  const testCases: Array<{ name: string; data: Buffer; shouldPass: boolean }> =
    [
      {
        name: "Too short data",
        data: Buffer.from(
          "VOYA\x01\x00\x00\x00\x0a\x00\x00\x00\x00\x20",
          "binary"
        ),
        shouldPass: false,
      },
      {
        name: "Valid minimal index",
        data: Buffer.concat([
          Buffer.from("VOYA", "binary"), // Header
          Buffer.from([0x01, 0x00, 0x00, 0x00]), // File version
          Buffer.from([0x0a, 0x00, 0x00, 0x00]), // Number of dimensions (10)
          Buffer.from([0x00]), // Space type
          Buffer.from([0x20]), // Storage data type
          Buffer.allocUnsafe(4).fill(0), // maximum norm (float)
          Buffer.from([0x00]), // Use order-preserving transform
          Buffer.allocUnsafe(8).fill(0), // offsetLevel0_
          Buffer.from([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), // max_elements_
          Buffer.from([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), // cur_element_count
          Buffer.from([0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), // size_data_per_element_
          Buffer.allocUnsafe(8).fill(0), // label_offset_
          Buffer.allocUnsafe(8).fill(0), // offsetData_
          Buffer.allocUnsafe(4).fill(0), // maxlevel_
          Buffer.allocUnsafe(4).fill(0), // enterpoint_node_
          Buffer.allocUnsafe(8).fill(0), // maxM_
          Buffer.allocUnsafe(8).fill(0), // maxM0_
          Buffer.allocUnsafe(8).fill(0), // M_
          Buffer.allocUnsafe(8).fill(0), // mult_
          Buffer.allocUnsafe(8).fill(0), // ef_construction_
          Buffer.allocUnsafe(52).fill(0), // one vector
          Buffer.allocUnsafe(4).fill(0), // one linklist
        ]),
        shouldPass: true,
      },
      {
        name: "Corrupted offset",
        data: Buffer.concat([
          Buffer.from("VOYA", "binary"),
          Buffer.from([0x01, 0x00, 0x00, 0x00]),
          Buffer.from([0x0a, 0x00, 0x00, 0x00]),
          Buffer.from([0x00]),
          Buffer.from([0x20]),
          Buffer.allocUnsafe(4).fill(0),
          Buffer.from([0x00]),
          Buffer.from([0x00, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00]), // Bad offsetLevel0_
          Buffer.from([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
          Buffer.from([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
          Buffer.from([0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
          Buffer.allocUnsafe(8).fill(0),
          Buffer.allocUnsafe(8).fill(0),
          Buffer.allocUnsafe(4).fill(0),
          Buffer.allocUnsafe(4).fill(0),
          Buffer.allocUnsafe(8).fill(0),
          Buffer.allocUnsafe(8).fill(0),
          Buffer.allocUnsafe(8).fill(0),
          Buffer.allocUnsafe(8).fill(0),
          Buffer.allocUnsafe(8).fill(0),
          Buffer.allocUnsafe(52).fill(0),
          Buffer.allocUnsafe(4).fill(0),
        ]),
        shouldPass: false,
      },
    ];

  let allPassed = true;
  for (const testCase of testCases) {
    const testName = `Load invalid data: ${testCase.name}`;
    try {
      if (testCase.shouldPass) {
        const index = Index.fromBuffer(testCase.data);
        if (index.ids.length !== 1) {
          throw new Error(`Expected length 1, got ${index.ids.length}`);
        }
        const vector = index.getVector(0);
        const expectedZeros = new Array(index.numDimensions).fill(0);
        assertArrayClose(
          vector,
          expectedZeros,
          0.01,
          "Vector should be all zeros"
        );
        console.log(`✓ ${testName}`);
      } else {
        let threwError = false;
        try {
          const index = Index.fromBuffer(testCase.data);
          // Try to query to ensure we don't segfault
          for (const id of index.ids) {
            index.query(index.getVector(id));
          }
        } catch {
          threwError = true;
        }

        if (!threwError) {
          throw new Error("Expected an error to be thrown");
        }
        console.log(`✓ ${testName}`);
      }
    } catch (error) {
      console.error(`✗ ${testName}`);
      console.error(
        `  Error: ${error instanceof Error ? error.message : String(error)}`
      );
      allPassed = false;
    }
  }
  return allPassed;
}

// Fuzz testing with random data
function testFuzz(
  seed: number,
  withValidHeader: boolean,
  offsetLevel0?: number
): boolean {
  const fuzzTestName = `Fuzz test: seed=${seed}, validHeader=${withValidHeader}, offset=${
    offsetLevel0 ?? "random"
  }`;

  try {
    // Simple deterministic random number generator
    let randomState = seed;
    const random = () => {
      randomState = (randomState * 1103515245 + 12345) & 0x7fffffff;
      return randomState / 0x7fffffff;
    };

    // Create random data buffer
    const numBytes = Math.floor(random() * 10000) + 100;
    const randomData = Buffer.allocUnsafe(numBytes);
    for (let i = 0; i < numBytes; i++) {
      randomData[i] = Math.floor(random() * 256);
    }

    if (withValidHeader) {
      const header = Buffer.from("VOYA", "binary");
      header.copy(randomData, 0);

      const version = Buffer.from([0x01, 0x00, 0x00, 0x00]);
      version.copy(randomData, 4);

      const dims = Buffer.from([0x0a, 0x00, 0x00, 0x00]);
      dims.copy(randomData, 8);

      const spaceType = Buffer.from([0x00]);
      spaceType.copy(randomData, 12);

      const storageType = Buffer.from([0x20]);
      storageType.copy(randomData, 13);

      if (offsetLevel0 !== undefined && randomData.length >= 22) {
        const offsetBuf = Buffer.allocUnsafe(8);
        offsetBuf.writeBigUInt64LE(BigInt(offsetLevel0), 0);
        offsetBuf.copy(randomData, 18);
      }
    }

    let threwError = false;
    try {
      Index.fromBuffer(randomData);
    } catch (error) {
      threwError = true;
    }

    if (!threwError) {
      throw new Error("Expected an error to be thrown for random data");
    }

    console.log(`✓ ${fuzzTestName}`);
    return true;
  } catch (error) {
    console.error(`✗ ${fuzzTestName}`);
    console.error(
      `  Error: ${error instanceof Error ? error.message : String(error)}`
    );
    return false;
  }
}

// Main test runner
export default function runAllTests(): boolean {
  console.log("Running Load Indices Tests...");
  console.log("=".repeat(70));

  let totalTests = 0;
  let passedTests = 0;

  // Test V0 indices
  console.log("\nTesting V0 indices...");
  const v0Files = getIndexFiles("v0");
  for (const indexFile of v0Files) {
    // Test loading from file
    totalTests++;
    if (testLoadV0Index(indexFile, false)) passedTests++;

    // Test loading from buffer
    totalTests++;
    if (testLoadV0Index(indexFile, true)) passedTests++;
  }

  // Test V1 indices (can be loaded with or without parameters)
  console.log("\nTesting V1 indices...");
  const v1Files = getIndexFiles("v1");
  for (const indexFile of v1Files) {
    // Test loading from file
    totalTests++;
    if (testLoadV1Index(indexFile, false)) passedTests++;

    // Test loading from buffer
    totalTests++;
    if (testLoadV1Index(indexFile, true)) passedTests++;
  }

  // Test V1 parameter validation
  console.log("\nTesting V1 parameter validation...");
  for (const indexFile of v1Files) {
    totalTests++;
    if (testV1IndicesMustMatchParameters(indexFile, false)) passedTests++;
    totalTests++;

    if (testV1IndicesMustMatchParameters(indexFile, true)) passedTests++;
  }

  // Test invalid data handling
  console.log("\nTesting invalid data handling...");
  totalTests++;
  if (testLoadingInvalidDataCannotCrash()) passedTests++;

  // Fuzz testing (limited subset for reasonable test time)
  console.log("\nRunning fuzz tests (limited)...");
  const fuzzTests = [
    { seed: 42, withValidHeader: true, offsetLevel0: 500000 },
    { seed: 123, withValidHeader: true, offsetLevel0: undefined },
    { seed: 456, withValidHeader: false, offsetLevel0: undefined },
    { seed: 789, withValidHeader: true, offsetLevel0: 1000000 },
    { seed: 1011, withValidHeader: false, offsetLevel0: undefined },
  ];

  for (const test of fuzzTests) {
    totalTests++;
    if (testFuzz(test.seed, test.withValidHeader, test.offsetLevel0))
      passedTests++;
  }

  const failed = totalTests - passedTests;

  console.log("\n=== Load Indices Test Summary ===");
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${failed}`);
  console.log(`Total: ${totalTests}`);

  if (failed === 0) {
    console.log("✓ All load indices tests passed!");
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
