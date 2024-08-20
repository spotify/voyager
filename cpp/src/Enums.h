#pragma once

#include <string>

/**
 * The space (i.e. distance metric) to use for searching.
 */
enum SpaceType : unsigned char {
  Euclidean = 0,
  InnerProduct = 1,
  Cosine = 2,
};

/**
 * The datatype used to use when storing vectors on disk.
 * Affects precision and memory usage.
 */
enum class StorageDataType : unsigned char {
  Float8 = 1 << 4,
  Float32 = 2 << 4,

  // An 8-bit floating point format that uses
  // four bits for exponent, 3 bits for mantissa,
  // allowing representation of values from 2e-9 to 448.
  E4M3 = 3 << 4,
};

inline const std::string toString(StorageDataType sdt) {
  switch (sdt) {
  case StorageDataType::Float8:
    return "Float8";
  case StorageDataType::Float32:
    return "Float32";
  case StorageDataType::E4M3:
    return "E4M3";
  default:
    return "Unknown storage data type (value " + std::to_string((int)sdt) + ")";
  }
}

inline const std::string toString(SpaceType space) {
  switch (space) {
  case SpaceType::Euclidean:
    return "Euclidean";
  case SpaceType::Cosine:
    return "Cosine";
  case SpaceType::InnerProduct:
    return "InnerProduct";
  default:
    return "Unknown space type (value " + std::to_string((int)space) + ")";
  }
}

std::ostream &operator<<(std::ostream &os, const SpaceType space) {
  os << toString(space);
  return os;
}

std::ostream &operator<<(std::ostream &os, const StorageDataType sdt) {
  os << toString(sdt);
  return os;
}
