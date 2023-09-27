#pragma once

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
