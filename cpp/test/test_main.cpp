#include "doctest.h"

#include "TypedIndex.h"
#include "test_utils.cpp"
#include <tuple>
#include <type_traits>

template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
void testIndexProperties(TypedIndex<dist_t, data_t, scalefactor> &index,
                         SpaceType spaceType, int numDimensions,
                         StorageDataType storageType) {
  REQUIRE(toString(index.getSpace()) == toString(spaceType));
  REQUIRE(index.getNumDimensions() == numDimensions);
  REQUIRE(toString(index.getStorageDataType()) == toString(storageType));
}

/**
 * Test the query method of the index. The index is populated with random
 * vectors, and then queried with the same vectors. For Float32/Float8 Euclidean
 * and cosine queries, require 99% matching IDs and correct returned distances.
 * Rare search misses are accepted; numerical tolerances remain unchanged.
 */
template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
void testQuery(TypedIndex<dist_t, data_t, scalefactor> &index, int numVectors,
               int numDimensions, SpaceType spaceType,
               StorageDataType storageType, bool testSingleVectorMethod,
               float precisionTolerance, int k) {
  /**
   * Create test data and ids. If we are using Float8 or E4M3 storage, quantize
   * the vector values, if we are using Float32 storage, keep the float values
   * as-is. We want to match the storage type use case with the input data.
   */
  std::vector<std::vector<float>> inputData;
  if (storageType == StorageDataType::Float8 ||
      storageType == StorageDataType::E4M3) {
    inputData = randomQuantizedVectors(numVectors, numDimensions);
  } else if (storageType == StorageDataType::Float32) {
    inputData = randomVectors(numVectors, numDimensions);
  }
  std::vector<hnswlib::labeltype> ids(numVectors);
  for (int i = 0; i < numVectors; i++) {
    ids[i] = i;
  }

  // add items to index
  if (testSingleVectorMethod == true) {
    for (auto id : ids) {
      index.addItem(inputData[id], id);
    }
  } else {
    index.addItems(inputData, ids, -1);
  }

  // Allow up to 1% misses from approximate search, including unreachable nodes.
  constexpr double minimumRecall = 0.99;
  // Preserve existing exclusions where quantization or inner product can change
  // which vector is the nearest neighbor to the original query.
  const bool checkRecall = storageType != StorageDataType::E4M3 &&
                           spaceType != SpaceType::InnerProduct;
  // A missed self-match must still report the distance to its returned vector.
  auto checkResult = [&](int queryId, hnswlib::labeltype label,
                         float distance) {
    REQUIRE(label < inputData.size());
    REQUIRE(std::isfinite(distance));
    if (checkRecall) {
      const float expected =
          index.getDistance(inputData[queryId], inputData[label]);
      REQUIRE(std::abs(distance - expected) <= precisionTolerance);
    }
  };

  // Use the single-query interface (query with a single target vector)
  for (long queryEf = 100; queryEf <= numVectors; queryEf *= 10) {
    int matches = 0;
    for (int i = 0; i < numVectors; i++) {

      /**
       * Use the raw inputData as target vectors for querying. We don't use the
       * index data because once data has been added to the index, the model can
       * change the "ground truth" by changing the data format.
       */
      auto targetVector = inputData[i];
      auto nearestNeighbor = index.query(targetVector, k, queryEf);

      auto labels = std::get<0>(nearestNeighbor);
      auto distances = std::get<1>(nearestNeighbor);
      REQUIRE(labels.size() == k);
      REQUIRE(distances.size() == k);

      checkResult(i, labels[0], distances[0]);
      // Count exact ID matches, then enforce the budget across all queries.
      matches += labels[0] == static_cast<hnswlib::labeltype>(i);
    }
    if (checkRecall) {
      CAPTURE(matches);
      REQUIRE(static_cast<double>(matches) / numVectors >= minimumRecall);
    }
  }

  // Use the bulk-query interface  (query with multiple target vectors at once)
  for (long queryEf = 100; queryEf <= numVectors; queryEf *= 10) {
    auto nearestNeighbors = index.query(
        inputData, /* k= */ k, /* numThreads= */ -1, /* queryEf= */ queryEf);
    NDArray<hnswlib::labeltype, 2> labels = std::get<0>(nearestNeighbors);
    NDArray<dist_t, 2> distances = std::get<1>(nearestNeighbors);
    REQUIRE(labels.shape[0] == numVectors);
    REQUIRE(labels.shape[1] == k);
    REQUIRE(distances.shape[0] == numVectors);
    REQUIRE(distances.shape[1] == k);

    int matches = 0;
    for (int i = 0; i < numVectors; i++) {
      auto label = labels.data[i];
      auto distance = distances.data[i];

      checkResult(i, label, distance);
      matches += label == static_cast<hnswlib::labeltype>(i);
    }
    // Enforce the same recall budget independently for bulk queries.
    if (checkRecall) {
      CAPTURE(matches);
      REQUIRE(static_cast<double>(matches) / numVectors >= minimumRecall);
    }
  }
}

TEST_CASE(
    "Randomized queries retain high recall and accurate returned distances") {
  std::unordered_map<StorageDataType, float> PRECISION_TOLERANCE_PER_DATA_TYPE =
      {{StorageDataType::Float32, 0.00001f},
       {StorageDataType::Float8, 0.10f},
       {StorageDataType::E4M3, 0.20f}};
  std::vector<SpaceType> spaceTypesSet = {
      SpaceType::Euclidean, SpaceType::InnerProduct, SpaceType::Cosine};
  std::vector<int> numDimensionsSet = {16};
  std::vector<int> numVectorsSet = {500};
  std::vector<StorageDataType> storageTypesSet = {
      StorageDataType::Float8, StorageDataType::Float32, StorageDataType::E4M3};
  std::vector<bool> testSingleVectorMethods = {true, false};
  int k = 1;

  for (auto spaceType : spaceTypesSet) {
    for (auto storageType : storageTypesSet) {
      for (auto numDimensions : numDimensionsSet) {
        for (auto numVectors : numVectorsSet) {
          for (auto testSingleVectorMethod : testSingleVectorMethods) {

            SUBCASE("Test instantiation ") {
              CAPTURE(spaceType);
              CAPTURE(numDimensions);
              CAPTURE(numVectors);
              CAPTURE(storageType);
              CAPTURE(testSingleVectorMethod);

              if (storageType == StorageDataType::Float8) {
                auto index = TypedIndex<float, int8_t, std::ratio<1, 127>>(
                    spaceType, numDimensions);
                testIndexProperties(index, spaceType, numDimensions,
                                    storageType);
                testQuery(index, numVectors, numDimensions, spaceType,
                          storageType, testSingleVectorMethod,
                          PRECISION_TOLERANCE_PER_DATA_TYPE[storageType], k);
              } else if (storageType == StorageDataType::Float32) {
                auto index = TypedIndex<float>(spaceType, numDimensions);
                testIndexProperties(index, spaceType, numDimensions,
                                    storageType);
                testQuery(index, numVectors, numDimensions, spaceType,
                          storageType, testSingleVectorMethod,
                          PRECISION_TOLERANCE_PER_DATA_TYPE[storageType], k);
              } else if (storageType == StorageDataType::E4M3) {
                auto index = TypedIndex<float, E4M3>(spaceType, numDimensions);
                testIndexProperties(index, spaceType, numDimensions,
                                    storageType);
                testQuery(index, numVectors, numDimensions, spaceType,
                          storageType, testSingleVectorMethod,
                          PRECISION_TOLERANCE_PER_DATA_TYPE[storageType], k);
              }
            }
          }
        }
      }
    }
  }
}

template <typename data_t, typename scalefactor = std::ratio<1, 1>>
void testConnectedIndex(SpaceType space) {
  // This small graph must return every self-match and every stored ID.
  TypedIndex<float, data_t, scalefactor> index(space, 4, 20);
  const std::vector<std::vector<float>> vectors = {
      {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  const std::vector<hnswlib::labeltype> ids = {10, 20, 30, 40};
  index.addItems(vectors, ids, 1);
  auto batch = index.query(vectors, 1, 1);
  REQUIRE(std::get<0>(batch).data == ids);
  REQUIRE(std::get<1>(batch).data == std::vector<float>(4, 0));
  for (size_t i = 0; i < vectors.size(); i++) {
    auto single = index.query(vectors[i], 1);
    REQUIRE(std::get<0>(single) == std::vector<hnswlib::labeltype>{ids[i]});
    REQUIRE(std::get<1>(single) == std::vector<float>{0});
    auto all = index.query(vectors[i], ids.size());
    auto labels = std::get<0>(all);
    REQUIRE(labels[0] == ids[i]);
    std::sort(labels.begin(), labels.end());
    REQUIRE(labels == ids);
    const float distance = space == SpaceType::Euclidean ? 2 : 1;
    const std::vector<float> expected = {0, distance, distance, distance};
    REQUIRE(std::get<1>(all) == expected);
  }
}

TEST_CASE("Connected four-vector graphs retain exact and exhaustive queries") {
  // Exactly representable basis vectors fit within every node's neighbor limit.
  for (auto space :
       {SpaceType::Euclidean, SpaceType::Cosine, SpaceType::InnerProduct}) {
    CAPTURE(space);
    testConnectedIndex<float>(space);
    testConnectedIndex<int8_t, std::ratio<1, 127>>(space);
  }
}

TEST_CASE(
    "Test vectorsToNDArray converts 2D vector of float to NDArray<float,2>") {
  std::vector<std::vector<float>> vectors = {{1.0f, 2.0f, 3.0f, 4.0f},
                                             {5.0f, 6.0f, 7.0f, 8.0f},
                                             {9.0f, 10.0f, 11.0f, 12.0f}};
  NDArray<float, 2> ndArray = vectorsToNDArray(vectors);
  REQUIRE(ndArray.shape.size() == 2);
  REQUIRE(ndArray.shape[0] == 3);
  REQUIRE(ndArray.shape[1] == 4);
  REQUIRE(ndArray.data.size() == 12);
  REQUIRE(ndArray.data[0] == 1.0f);
  REQUIRE(ndArray.data[1] == 2.0f);
  REQUIRE(ndArray.data[2] == 3.0f);
  REQUIRE(ndArray.data[3] == 4.0f);
  REQUIRE(ndArray.data[4] == 5.0f);
  REQUIRE(ndArray.data[5] == 6.0f);
  REQUIRE(ndArray.data[6] == 7.0f);
  REQUIRE(ndArray.data[7] == 8.0f);
  REQUIRE(ndArray.data[8] == 9.0f);
  REQUIRE(ndArray.data[9] == 10.0f);
  REQUIRE(ndArray.data[10] == 11.0f);
  REQUIRE(ndArray.data[11] == 12.0f);
  REQUIRE(*ndArray[0] == 1.0f);
  REQUIRE(*ndArray[1] == 5.0f);
  REQUIRE(*ndArray[2] == 9.0f);
}

TEST_CASE(
    "Test vectorsToNDArray throws error if vectors are not of the same size") {
  std::vector<std::vector<float>> vectors1 = {{1.0f, 2.0f, 3.0f, 4.0f},
                                              {5.0f, 6.0f, 7.0f},
                                              {9.0f, 10.0f, 11.0f, 12.0f}};
  REQUIRE_THROWS_AS(vectorsToNDArray(vectors1), std::invalid_argument);

  std::vector<std::vector<float>> vectors2 = {
      {1.0f}, {5.0f, 6.0f, 7.0f}, {9.0f, 10.0f, 11.0f}};
  REQUIRE_THROWS_AS(vectorsToNDArray(vectors2), std::invalid_argument);
}
