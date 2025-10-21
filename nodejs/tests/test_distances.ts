import { Index, Space, StorageDataType } from "../src/voyager-node.ts";

// Helper functions for distance calculations
function normalized(vec: number[]): number[] {
  const sum = vec.reduce((acc, val) => acc + val * val, 0);
  const norm = Math.sqrt(sum) + 1e-30;
  return vec.map((v) => v / norm);
}

function innerProductDistance(a: number[], b: number[]): number {
  const dotProduct = a.reduce((acc, val, i) => acc + val * b[i], 0);
  return 1.0 - dotProduct;
}

function cosineDistance(a: number[], b: number[]): number {
  return innerProductDistance(normalized(a), normalized(b));
}

function l2Square(a: number[], b: number[]): number {
  return a.reduce((acc, val, i) => acc + Math.pow(val - b[i], 2), 0);
}

function quantizeToFloat8(vec: number[]): number[] {
  return vec.map((v) => {
    const quantized = Math.floor(v * 127); // replicating Python's astype(np.int8)
    // wrapping around is not an issue here since input is [0,1)
    return quantized / 127.0;
  });
}

function quantizeToE4M3(vec: number[]): number[] {
  // E4M3 quantization - simplified approximation
  // The actual E4M3 conversion would require the C++ E4M3 class
  // For testing purposes, we'll clamp to the E4M3 range and reduce precision
  return vec.map((v) => {
    const clamped = Math.max(-448, Math.min(448, v));
    // Simulate reduced precision by rounding to fewer significant digits
    return parseFloat(clamped.toPrecision(3));
  });
}

// Test parameters
const dimensions = [1, 2, 5, 7, 13, 40, 100];

const spaces = [
  { space: Space.Euclidean, name: "Euclidean" },
  { space: Space.Cosine, name: "Cosine" },
  { space: Space.InnerProduct, name: "InnerProduct" },
];

const storageTypes = [
  { type: StorageDataType.Float32, tolerance: 1e-5, name: "Float32" },
  { type: StorageDataType.Float8, tolerance: 0.1, name: "Float8" },
  { type: StorageDataType.E4M3, tolerance: 0.2, name: "E4M3" },
];

function generateRandomVector(size: number): number[] {
  return Array.from({ length: size }, () => Math.random());
}
function testDistance(
  dimensions: number,
  space: Space,
  spaceName: string,
  storageDataType: StorageDataType,
  storageTypeName: string,
  tolerance: number
): boolean {
  const testName = `Distance test: ${dimensions}D, ${spaceName}, ${storageTypeName}`;

  try {
    const index = new Index({
      space,
      numDimensions: dimensions,
      storageDataType,
    });

    let a = generateRandomVector(dimensions);
    let b = generateRandomVector(dimensions);

    const actual = index.getDistance(a, b);

    // Apply transformations as needed
    if (space === Space.Cosine) {
      a = normalized(a);
      b = normalized(b);
    }

    if (storageDataType === StorageDataType.Float8) {
      a = quantizeToFloat8(a);
      b = quantizeToFloat8(b);
    } else if (storageDataType === StorageDataType.E4M3) {
      a = quantizeToE4M3(a);
      b = quantizeToE4M3(b);
    }

    let expected: number;
    if (space === Space.Cosine) {
      // Don't re-normalize after quantization
      expected = innerProductDistance(a, b);
    } else if (space === Space.InnerProduct) {
      expected = innerProductDistance(a, b);
    } else if (space === Space.Euclidean) {
      expected = l2Square(a, b);
    } else {
      throw new Error(`Unknown space type: ${space}`);
    }

    const diff = Math.abs(actual - expected);
    if (diff >= tolerance) {
      console.error(`✗ ${testName}`);
      console.error(
        `  Expected: ${expected}, Got: ${actual}, Diff: ${diff}, Tolerance: ${tolerance}`
      );
      return false;
    }

    console.log(`✓ ${testName}`);
    return true;
  } catch (error) {
    console.error(`✗ ${testName}`);
    console.error(`Error: ${error}`);
    return false;
  }
}

export default function runAllTests(): boolean {
  console.log("Running distance calculation tests...\n");
  let totalTests = 0;
  let passedTests = 0;

  for (const dim of dimensions) {
    for (const { space, name: spaceName } of spaces) {
      for (const { type, tolerance, name: storageTypeName } of storageTypes) {
        totalTests++;
        if (
          testDistance(dim, space, spaceName, type, storageTypeName, tolerance)
        ) {
          passedTests++;
        }
      }
    }
  }

  const failed = totalTests - passedTests;
  console.log("\n=== Distance Test Summary ===");
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${failed}`);
  console.log(`Total: ${totalTests}`);

  if (failed === 0) {
    console.log("✓ All distance tests passed!");
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
