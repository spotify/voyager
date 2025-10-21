import runDistanceTests from "./test_distances.ts";
import runE4M3Tests from "./test_e4m3.ts";
import runIndexCreationTests from "./test_index_creation.ts";
import runIndexRecreationTests from "./test_index_recreation.ts";
import runLoadIndicesTests from "./test_load_indices.ts";
import process from "process";

async function main() {
  console.log("=".repeat(70));
  console.log("Voyager Node.js Bindings - Test Suite");
  console.log("=".repeat(70));
  console.log();

  let allPassed = true;
  const failedTests: string[] = [];
  try {
    console.log("Running Distance Calculation Tests...");
    console.log("=".repeat(70));
    runDistanceTests();
    console.log("✓ Distance tests passed");
  } catch (error) {
    console.error("✗ Distance tests failed with error:", error);
    console.log("+++++++++++++++");
    failedTests.push("Distance Calculation Tests");
    allPassed = false;
  }
  console.log();
  try {
    console.log("Running E4M3 Tests...");
    console.log("=".repeat(70));
    runE4M3Tests();
    console.log("✓ E4M3 tests passed");
  } catch (error) {
    console.error("✗ E4M3 tests failed with error:", error);
    failedTests.push("EM3 Tests");
    allPassed = false;
  }
  console.log();

  try {
    console.log("Running Index Creation Tests...");
    console.log("=".repeat(70));
    runIndexCreationTests();
    console.log("✓ Index creation tests passed");
  } catch (error) {
    console.error("✗ Index creation tests failed with error:", error);
    failedTests.push("Index Creation Tests");
    allPassed = false;
  }
  console.log();
  try {
    console.log("Running Index Recreation Tests...");
    console.log("=".repeat(70));
    runIndexRecreationTests();
    console.log("✓ Index recreation tests passed");
  } catch (error) {
    console.error("✗ Index recreation tests failed with error:", error);
    failedTests.push("Index Recreation Tests");
    allPassed = false;
  }
  console.log();

  try {
    console.log("Running Load Indices Tests...");
    console.log("=".repeat(70));
    runLoadIndicesTests();
    console.log(" Load indices tests passed");
  } catch (error) {
    console.error("✗ Load indices tests failed with error:", error);
    failedTests.push("Load Indices Tests");
    allPassed = false;
  }
  console.log();
  console.log("=".repeat(70));
  console.log("Test Suite Summary");
  console.log("=".repeat(70));
  if (allPassed) {
    console.log("✓ All tests passed!");
  } else {
    console.log(`✗ ${failedTests.length} test suite(s) failed:`);
    failedTests.forEach((test) => console.log(`  - ${test}`));
    process.exit(1);
  }
}

main();
