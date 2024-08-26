#include <random>
#include <vector>

#include "array_utils.h"

NDArray<float, 2> randomQuantizedVectorsNDArray(int numVectors,
                                                int dimensions) {
  NDArray<float, 2> vectors = NDArray<float, 2>({numVectors, dimensions});

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1.0);

  int numElements = numVectors * dimensions;
  for (int i = 0; i < numElements; ++i) {
    vectors.data[i] = static_cast<int>(((dis(gen) * 2 - 1) * 10.0f)) / 10.0f;
  }

  return vectors;
}

NDArray<float, 2> randomVectorsNDArray(int numVectors, int dimensions) {
  NDArray<float, 2> vectors = NDArray<float, 2>({numVectors, dimensions});

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1.0);

  int numElements = numVectors * dimensions;
  for (int i = 0; i < numElements; ++i) {
    vectors.data[i] = static_cast<float>(dis(gen)) * 2 - 1;
  }

  return vectors;
}

std::vector<std::vector<float>> randomQuantizedVectors(int numVectors,
                                                       int dimensions) {
  std::vector<std::vector<float>> vectors(numVectors,
                                          std::vector<float>(dimensions));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1.0);

  for (int i = 0; i < numVectors; ++i) {
    for (int j = 0; j < dimensions; ++j) {
      vectors[i][j] = static_cast<int>(((dis(gen) * 2 - 1) * 10.0f)) / 10.0f;
    }
  }

  return vectors;
}

std::vector<std::vector<float>> randomVectors(int numVectors, int dimensions) {
  std::vector<std::vector<float>> vectors(numVectors,
                                          std::vector<float>(dimensions));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1.0);

  for (int i = 0; i < numVectors; ++i) {
    for (int j = 0; j < dimensions; ++j) {
      vectors[i][j] = static_cast<float>(dis(gen)) * 2 - 1;
    }
  }

  return vectors;
}
