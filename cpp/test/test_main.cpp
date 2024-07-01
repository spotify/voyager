#include "doctest.h"

#include "TypedIndex.h"
#include <tuple>
#include <type_traits>

template <typename dist_t, typename data_t = dist_t, typename scalefactor = std::ratio<1, 1>>
void testCombination(TypedIndex<dist_t, data_t, scalefactor> &index, SpaceType spaceType, int numDimensions,
                     StorageDataType storageType) {
  CHECK(toString(index.getSpace()) == toString(spaceType));
  CHECK(index.getNumDimensions() == numDimensions);
  CHECK(toString(index.getStorageDataType()) == toString(storageType));
}

TEST_CASE("Test combinations of different instantiations and sizes") {
  std::vector<SpaceType> spaceTypesSet = {SpaceType::Euclidean, SpaceType::InnerProduct};
  std::vector<int> numDimensionsSet = {4, 16, 128, 1024};
  std::vector<int> numElementsSet = {100, 1000, 100000};
  std::vector<StorageDataType> storageTypesSet = {StorageDataType::Float8, StorageDataType::Float32,
                                                  StorageDataType::E4M3};

  for (auto spaceType : spaceTypesSet) {
    for (auto numDimensions : numDimensionsSet) {
      for (auto numElements : numElementsSet) {
        for (auto storageType : storageTypesSet) {
          SUBCASE("Combination test") {
            CAPTURE(spaceType);
            CAPTURE(numDimensions);
            CAPTURE(numElements);
            CAPTURE(storageType);

            if (storageType == StorageDataType::Float8) {
              auto index = TypedIndex<float, int8_t, std::ratio<1, 127>>(spaceType, numDimensions);
              testCombination(index, spaceType, numDimensions, storageType);
            } else if (storageType == StorageDataType::Float32) {
              auto index = TypedIndex<float>(spaceType, numDimensions);
              testCombination(index, spaceType, numDimensions, storageType);
            } else if (storageType == StorageDataType::E4M3) {
              auto index = TypedIndex<float, E4M3>(spaceType, numDimensions);
              testCombination(index, spaceType, 20, storageType);
            }
          }
        }
      }
    }
  }
}
