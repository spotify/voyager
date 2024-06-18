#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "TypedIndex.h"
#include <tuple>
#include <type_traits>

#include "doctest.h"

TEST_CASE("Basic Init") {

  for (auto [space_type, num_dimensions, num_elements] : {
           std::make_tuple(SpaceType::Euclidean, 16, 100),
           std::make_tuple(SpaceType::Euclidean, 128, 100),
           std::make_tuple(SpaceType::InnerProduct, 256, 100),
           std::make_tuple(SpaceType::InnerProduct, 4, 100),
           std::make_tuple(SpaceType::InnerProduct, 4, 1000),
       }) {
    SUBCASE("Jawn") {
      CAPTURE(std::make_tuple(space_type, num_dimensions, num_elements));
      auto index = TypedIndex<float>(space_type, num_dimensions);
      CHECK(toString(index.getSpace()) == toString(space_type));
      CHECK(toString(index.getStorageDataType()) == toString(StorageDataType::Float32));
      CHECK(index.getNumDimensions() == num_dimensions);
    }
  }
}
