#include "TypedIndex.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>

// TODO: Extract data generation as a function or as a Catch2 Generator

// template <typename T> struct AllCloseMatcher : Catch::Matchers::MatcherGenericBase {
//   AllCloseMatcher(const std::vector<T> &a, const float rtol = 1e-7, const float atol = 0)
//       : a_(a), rtol_(rtol), atol_(atol) {}

//   bool match(const std::vector<T> &b) const {
//     // Could use std::reduce, but early return is most likely faster
//     if (a_.size() != b.size()) {
//       return false;
//     }
//     // TODO: Replace with Ranges https://en.cppreference.com/w/cpp/ranges
//     for (int i = 0; i < a_.size(); ++i) {
//       if (!(std::fabs(a_[i] - b[i]) <= (atol_ + rtol_ * std::fabs(a_[i])))) {
//         return false;
//       }
//     }
//     return true;
//   }

//   std::string describe() const override { return "IsClose"; }

// private:
//   const std::vector<T> &a_;
//   const float atol_;
//   const float rtol_;
// };

// template <typename T>
// auto AllClose(const std::vector<T> a, const float rtol = 1e-7, const float atol = 0)
//     -> AllCloseMatcher<T> {
//   return AllCloseMatcher{a, rtol, atol};
// }

// template <typename T> std::vector<T> flattenNDArray(NDArray<T, 2> &arr) {
//   std::vector<T> res(arr.shape[0]);
//   for (auto i = 0; i < arr.shape[0]; ++i) {
//     res[i] = arr[i][0];
//   }
//   return res;
// };

TEST_CASE("Basic init") {
  auto space = GENERATE(SpaceType::Euclidean, SpaceType::InnerProduct);
  auto num_dimensions = GENERATE(4, 16, 128, 256);
  auto num_elements = GENERATE(100, 1000);

  SECTION("(num_dimensions, num_elements, space): (" + std::to_string(num_dimensions) + "," +
          std::to_string(num_elements) + "," + std::to_string(space) + ")") {
    auto index = TypedIndex<float>(space, num_dimensions);
    REQUIRE(index.getSpace() == space);
    REQUIRE(index.getStorageDataType() == StorageDataType::Float32);
    REQUIRE(index.getNumDimensions() == num_dimensions);
  }
}

// // dist_t, data_t, scalefactor, tolerance

// TEMPLATE_TEST_CASE("create_and_query",
//                    "[index_creation]",
//                    (std::tuple<float, struct E4M3, std::ratio<1, 1>>),
//                   //  (std::tuple<float, char, std::ratio<1, 127>>),
//                    (std::tuple<float, float, std::ratio<1, 1>>)) {
//   auto num_dimensions = GENERATE(4, 16, 128, 128, 256);
//   auto num_elements = GENERATE(100, 1000);
//   auto space = GENERATE(SpaceType::Euclidean, SpaceType::Cosine);

//   // It's a struggle to include these as std::ratio in the TEMPLATE test case so
//   // we'll set distance tolerance here.
//   float distance_tolerance = 0.0;
//   if (std::is_same<typename std::tuple_element<1, TestType>::type, struct E4M3>::value) {
//     distance_tolerance = 0.20;
//   } else if (std::is_same<typename std::tuple_element<1, TestType>::type, char>::value) {
//     distance_tolerance = 0.20;
//   } else if (std::is_same<typename std::tuple_element<1, TestType>::type, float>::value) {
//     distance_tolerance = 2e-6;
//   }

//   SECTION("(num_dimensions, num_elements, space): (" + std::to_string(num_dimensions) + "," +
//           std::to_string(num_elements) + "," + std::to_string(space) + ")") {

//     // Generate a 2D Matrix of test data
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dis(0.0, 1.0);
//     auto input_data = std::vector<float>(num_elements * num_dimensions);
//     std::generate(input_data.begin(), input_data.end(), [&dis, &gen]() {
//       float val = 2 * dis(gen) - 1;
//       if (std::is_same<typename std::tuple_element<1, TestType>::type, char>::value) {
//         val = std::round(val * 127.0f) / 127.0f;
//       }
//       return val;
//     });

//     auto input_array = NDArray<float, 2>(input_data, {num_elements, num_dimensions});

//     // Create Index
//     auto index = TypedIndex<typename std::tuple_element<0, TestType>::type,
//                             typename std::tuple_element<1, TestType>::type,
//                             typename std::tuple_element<2, TestType>::type>(
//         space, num_dimensions, 20, num_elements);

//     index.setEF(num_elements);
//     index.addItems(input_array);
//     SECTION("Multiple query interface") {
//       auto [labels, distances] = index.query(input_array);

//       if (!std::is_same<typename std::tuple_element<1, TestType>::type, float>::value) {
//         auto matches = 0;
//         // Could be std::reduce or std::accumulate
//         for (auto row = 0; row < num_elements; ++row) {
//           matches += labels[row][0] == row;
//         }
//         REQUIRE((double)matches / (double)num_elements > 0.5);
//       } else {
//         // Could be std::reduce or std::accumulate
//         std::vector<hnswlib::labeltype> expected(num_elements);
//         std::iota(expected.begin(), expected.end(), 0);
//         REQUIRE_THAT(flattenNDArray(labels), AllClose(expected));
//       }

//       REQUIRE_THAT(flattenNDArray(distances),
//                  AllClose(std::vector<float>(num_elements, 0.0), 1e-7, distance_tolerance));
//     }

//     SECTION("Single query interface") {
//       for (auto row = 0; row < num_elements; ++row) {
//         auto [labels, distances] =
//             index.query({input_array[row], input_array[row] + num_dimensions});
//         if (std::is_same<typename std::tuple_element<1, TestType>::type, float>::value) {
//           REQUIRE(labels[0] == row);
//         }
//         if(distances[0] >= distance_tolerance) {
//           float a = 0;
//         }
//         REQUIRE(distances[0] < distance_tolerance);
//       }
//     }

//     // SECTION("Saving an index") {
//     //   auto output_file = std::tmpfile();
//     //   index.saveIndex(std::make_shared<FileOutputStream>(output_file));
//     //   auto file_byte_count = std::ftell(output_file);
//     //   REQUIRE(file_byte_count > 0);
//     //   auto memory_output_stream = std::make_shared<MemoryOutputStream>();
//     //   index.saveIndex(memory_output_stream);
//     //   auto index_bytes = memory_output_stream->getValue().size();
//     //   REQUIRE(index_bytes > 0);
//     //   REQUIRE(file_byte_count == index_bytes);
//     // }
//   }
// }

// TEST_CASE("Spaces") {
//   auto [space, expected_distances] =
//       GENERATE(std::make_tuple<SpaceType, std::vector<float>>(SpaceType::Euclidean,
//                                                               {0.0, 1.0, 2.0, 2.0, 2.0}),
//                std::make_tuple<SpaceType, std::vector<float>>(SpaceType::InnerProduct,
//                                                               {-2.0, -1.0, 0.0, 0.0, 0.0}),
//                std::make_tuple<SpaceType, std::vector<float>>(
//                    SpaceType::Cosine, {0, 1.835e-1, 4.23e-1, 4.23e-1, 4.23e-1}));
//   auto right_dimension = GENERATE(Catch::Generators::range(1, 128, 3));
//   auto left_dimension = GENERATE(Catch::Generators::range(1, 32, 5));

//   auto num_dimensions = 3;
//   auto data = NDArray<float, 2>({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1}, {5, num_dimensions});

//   auto input_data = std::vector<float>();
//   for (int i = 0; i < data.shape[0]; ++i) {
//     std::vector<float> to_insert(left_dimension, 0);
//     std::vector<float> right(right_dimension, 0);
//     to_insert.insert(to_insert.end(), data[0], data[0] + data.shape[1]);
//     to_insert.insert(to_insert.end(), right.begin(), right.end());
//     input_data.insert(input_data.end(), to_insert.begin(), to_insert.end());
//   }

//   num_dimensions = right_dimension + left_dimension + data.shape[1];

//   auto data_2 = NDArray<float, 2>(input_data, {data.shape[0], num_dimensions});
//   auto index = TypedIndex<float>(space, num_dimensions, 16, 100);
//   index.setEF(10);
//   index.addItems(data_2);

//   auto [labels, distances] = index.query(
//       std::vector(data_2[data_2.shape[0] - 1], data_2[data_2.shape[0] - 1] + num_dimensions), 5);
//   REQUIRE_THAT(distances, AllClose(expected_distances, 1e-7, 1e-3));
// }

// TEST_CASE("Get Vectors") {
//   auto num_dimensions = GENERATE(4, 16, 128, 256);
//   auto num_elements = GENERATE(100, 1000);
//   auto space = GENERATE(SpaceType::Euclidean, SpaceType::InnerProduct);

//   // Generate a 2D Matrix of test data
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<float> dis(0.0, 1.0);
//   auto input_data = std::vector<float>(num_elements * num_dimensions);
//   std::generate(input_data.begin(), input_data.end(), [&dis, &gen]() { return 2 * dis(gen) - 1; });
//   auto input_array = NDArray<float, 2>(input_data, {num_elements, num_dimensions});

//   auto index = TypedIndex<float>(space, num_dimensions);
//   auto labels = std::vector<hnswlib::labeltype>(num_elements);
//   std::iota(labels.begin(), labels.end(), 0);

//   REQUIRE_THROWS(index.getVector(labels[0]));
//   index.addItems(input_array);

//   SECTION("Test single vector retrieval") {
//     for (auto i = 0; i < labels.size(); ++i) {
//       REQUIRE_THAT(index.getVector(labels[i]),
//                  AllClose(std::vector<float>(input_array[i], input_array[i] + num_dimensions)));
//     }
//   }

//   SECTION("Test all vectors retrieval") {
//     auto vectors = index.getVectors(labels);
//     for (auto i = 0; i < labels.size(); ++i) {
//       REQUIRE_THAT(std::vector<float>(vectors[i], vectors[i] + num_dimensions),
//                  AllClose(std::vector<float>(input_array[i], input_array[i] + num_dimensions)));
//     }
//   }
// }

// TEST_CASE("Query EF") {
//   auto space = GENERATE(SpaceType::Euclidean, SpaceType::InnerProduct);
//   auto [query_ef, rank_tolerance] =
//       GENERATE(std::make_tuple(1, 100), std::make_tuple(2, 75), std::make_tuple(100, 1));
//   auto num_dimensions = 32;
//   auto num_elements = 1000;

//   // Generate a 2D Matrix of test data
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<float> dis(0.0, 1.0);
//   auto input_data = std::vector<float>(num_elements * num_dimensions);
//   std::generate(input_data.begin(), input_data.end(), [&dis, &gen]() { return 2 * dis(gen) - 1; });
//   auto input_array = NDArray<float, 2>(input_data, {num_elements, num_dimensions});

//   auto index = TypedIndex<float>(space, num_dimensions, 20, num_elements);
//   index.setEF(num_elements);
//   index.addItems(input_array);

//   auto [closest_labels_per_vector, _] = index.query(input_array, num_elements, -1, num_elements);
//   SECTION("Multi query interface") {
//     auto [labels, _] = index.query(input_array, 1, -1, query_ef);
//     for (auto i = 0; i < labels.shape[0]; ++i) {
//       auto returned_label = labels[0][0];
//       // Consider doing this in a loop with an early break.
//       auto label_iter = std::find(closest_labels_per_vector[i],
//                                   closest_labels_per_vector[i] + closest_labels_per_vector.shape[1],
//                                   returned_label);
//       auto actual_rank = std::distance(closest_labels_per_vector[i], label_iter);
//       REQUIRE(actual_rank < rank_tolerance);
//     }
//   }

//   SECTION("Single query interface") {
//     for (auto i = 0; i < input_array.shape[0]; ++i) {
//       auto [returned_labels, _] =
//           index.query({input_data[i], input_data[i] + num_dimensions}, 1, query_ef);
//       auto returned_label = returned_labels[0];
//       auto label_iter = std::find(closest_labels_per_vector[i],
//                                   closest_labels_per_vector[i] + closest_labels_per_vector.shape[1],
//                                   returned_label);
//       auto actual_rank = std::distance(closest_labels_per_vector[i], label_iter);
//       REQUIRE(actual_rank < rank_tolerance);
//     }
//   }
// }