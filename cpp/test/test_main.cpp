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
 * vectors, and then queried with the same vectors. The expected result is that
 * each vector's nearest neighbor is itself and that the distance is zero
 * (allowing for some precision error based on the storage type).
 */
template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
void testQuery(TypedIndex<dist_t, data_t, scalefactor> &index, int numVectors,
               int numDimensions, SpaceType spaceType,
               StorageDataType storageType, bool testSingleVectorMethod,
               float precisionTolerance) {
  // create test data and ids
  std::vector<std::vector<float>> inputData =
      randomVectors(numVectors, numDimensions);
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

  int k = 1;
  float lowerBound = 0.0f - precisionTolerance;
  float upperBound = 0.0f + precisionTolerance;

  // Use the single-query interface (query with a single target vector)
  for (long queryEf = 100; queryEf <= numVectors; queryEf *= 10) {
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

      /**
       * E4M3 is too low precision for us to confidently assume that querying
       * with the unquantized (fp32) vector will return the quantized vector as
       * its NN InnerProduct will have negative distance to the closest item,
       * not zero
       */
      if (storageType != StorageDataType::E4M3 &&
          spaceType != SpaceType::InnerProduct) {
        REQUIRE(i == labels[0]);
        REQUIRE(distances[0] >= lowerBound);
        REQUIRE(distances[0] <= upperBound);
      }
    }
  }

  // Use the bulk-query interface  (query with multiple target vectors at once)
  for (long queryEf = 100; queryEf <= numVectors; queryEf *= 10) {
    for (int i = 0; i < numVectors; i++) {
      auto nearestNeighbors = index.query(
          inputData, /* k= */ 1, /* numThreads= */ -1, /* queryEf= */ queryEf);
      NDArray<hnswlib::labeltype, 2> labels = std::get<0>(nearestNeighbors);
      NDArray<dist_t, 2> distances = std::get<1>(nearestNeighbors);
      REQUIRE(labels.shape[0] == numVectors);
      REQUIRE(labels.shape[1] == k);
      REQUIRE(distances.shape[0] == numVectors);
      REQUIRE(distances.shape[1] == k);

      for (int i = 0; i < numVectors; i++) {
        auto label = labels.data[i];
        auto distance = distances.data[i];

        /**
         * E4M3 is too low precision for us to confidently assume that querying
         * with the unquantized (fp32) vector will return the quantized vector
         * as its NN InnerProduct will have negative distance to the closest
         * item, not zero
         */
        if (storageType != StorageDataType::E4M3 &&
            spaceType != SpaceType::InnerProduct) {
          REQUIRE(i == label);
          REQUIRE(distance >= lowerBound);
          REQUIRE(distance <= upperBound);
        }
      }
    }
  }
}

TEST_CASE("Test combinations of different instantiations. Test that each "
          "vector's NN is itself and distance is approximately zero.") {
  std::unordered_map<StorageDataType, float> PRECISION_TOLERANCE_PER_DATA_TYPE =
      {{StorageDataType::Float32, 0.00001f},
       {StorageDataType::Float8, 0.10f},
       {StorageDataType::E4M3, 0.20f}};
  std::vector<SpaceType> spaceTypesSet = {
      SpaceType::Euclidean, SpaceType::InnerProduct, SpaceType::Cosine};
  std::vector<int> numDimensionsSet = {32};
  std::vector<int> numVectorsSet = {500};
  std::vector<StorageDataType> storageTypesSet = {
      StorageDataType::Float8, StorageDataType::Float32, StorageDataType::E4M3};
  std::vector<bool> testSingleVectorMethods = {true, false};

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

              if (storageType == StorageDataType::Float8) {
                auto index = TypedIndex<float, int8_t, std::ratio<1, 127>>(
                    spaceType, numDimensions);
                testIndexProperties(index, spaceType, numDimensions,
                                    storageType);
                testQuery(index, numVectors, numDimensions, spaceType,
                          storageType, testSingleVectorMethod,
                          PRECISION_TOLERANCE_PER_DATA_TYPE[storageType]);
              } else if (storageType == StorageDataType::Float32) {
                auto index = TypedIndex<float>(spaceType, numDimensions);
                testIndexProperties(index, spaceType, numDimensions,
                                    storageType);
                testQuery(index, numVectors, numDimensions, spaceType,
                          storageType, testSingleVectorMethod,
                          PRECISION_TOLERANCE_PER_DATA_TYPE[storageType]);
              } else if (storageType == StorageDataType::E4M3) {
                auto index = TypedIndex<float, E4M3>(spaceType, numDimensions);
                testIndexProperties(index, spaceType, numDimensions,
                                    storageType);
                testQuery(index, numVectors, numDimensions, spaceType,
                          storageType, testSingleVectorMethod,
                          PRECISION_TOLERANCE_PER_DATA_TYPE[storageType]);
              }
            }
          }
        }
      }
    }
  }
}
