// Copyright 2026 Spotify AB
// Licensed under the Apache License, Version 2.0.
// macOS allocator attribution for BiMap, independently of the graph and Python.
// clang++ -std=c++17 -O2 -I cpp/src \
//   benchmarks/scripts/measure_bimap_memory.cpp -o /tmp/measure-bimap-memory

#include "BiMap.h"
#include <cstdio>
#include <cstdlib>
#include <malloc/malloc.h>
#include <new>
#include <type_traits>

// No allocations outside the measured interval are released during insertion.
// Count actual live malloc block bytes, including allocation size-class
// rounding.
static bool measuring = false;
static size_t liveBytes = 0;
static size_t liveBlocks = 0;

void *operator new(std::size_t size) {
  void *p = std::malloc(size ? size : 1);
  if (!p)
    throw std::bad_alloc();
  if (measuring) {
    liveBytes += malloc_size(p);
    ++liveBlocks;
  }
  return p;
}
void operator delete(void *p) noexcept {
  if (!p)
    return;
  if (measuring) {
    liveBytes -= malloc_size(p);
    --liveBlocks;
  }
  std::free(p);
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { ::operator delete(p); }

// Supports both the original owning reverse map and the pointer-based version.
template <typename T> const std::string &stringValue(const T &value) {
  if constexpr (std::is_pointer_v<T>)
    return *value;
  else
    return value;
}

std::vector<std::string> makeNames(size_t count, size_t length) {
  std::vector<std::string> inputs;
  inputs.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    std::string suffix = std::to_string(i);
    inputs.push_back("item:" + std::string(length - 5 - suffix.size(), '0') +
                     suffix);
  }
  return inputs;
}

void measure(size_t count, size_t length) {
  auto inputs = makeNames(count, length);
  voyager::BiMap map;
  liveBytes = liveBlocks = 0;
  measuring = true;
  for (size_t i = 0; i < count; ++i)
    map.insert(i, inputs[i]);
  measuring = false;

  using ReverseValue =
      typename std::decay_t<decltype(map.entries())>::mapped_type;
  constexpr size_t copies = std::is_pointer_v<ReverseValue> ? 1 : 2;
  size_t stringBytes = 0;
  for (const auto &entry : map.entries())
    stringBytes += malloc_size(stringValue(entry.second).data());
  stringBytes *= copies;
  std::printf("%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", count, length, copies,
              liveBytes, liveBlocks, count * length * copies, stringBytes,
              liveBytes - stringBytes);
}

int main() {
  std::puts(
      "items,name_bytes,copies,live_bytes,live_blocks,string_payload_bytes,"
      "string_allocation_bytes,node_and_bucket_bytes");
  measure(100000, 32);
  measure(100000, 128);
}
