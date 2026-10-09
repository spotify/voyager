/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 * --
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * -/-/-
 */

#pragma once

#include "StreamUtils.h"
#include "hnswlib.h"
#include <unordered_map>

namespace voyager {
// A bijection: names and labels are both unique. No binding owns a second copy.
class BiMap {
public:
  BiMap() = default;
  BiMap(const BiMap &other) {
    for (const auto &entry : other.byName)
      insert(entry.second, entry.first);
  }
  BiMap &operator=(const BiMap &other) {
    BiMap copy(other);
    *this = std::move(copy);
    return *this;
  }
  BiMap(BiMap &&) noexcept = default;
  BiMap &operator=(BiMap &&) noexcept = default;

  bool containsLabel(hnswlib::labeltype label) const {
    return byLabel.count(label);
  }
  bool contains(const std::string &name) const { return byName.count(name); }
  hnswlib::labeltype label(const std::string &name) const {
    auto found = byName.find(name);
    if (found == byName.end())
      throw std::out_of_range("Unknown string identifier: " + name);
    return found->second;
  }
  const std::string &name(hnswlib::labeltype label) const {
    auto found = byLabel.find(label);
    if (found == byLabel.end())
      throw std::out_of_range("No string identifier for label " +
                              std::to_string(label));
    return *found->second;
  }
  void insert(hnswlib::labeltype label, const std::string &name) {
    if (byName.count(name) || byLabel.count(label))
      throw std::domain_error(
          "Duplicate string identifier or label in Voyager mapping.");
    auto owner = byName.emplace(name, label).first;
    try {
      byLabel.emplace(label, &owner->first);
    } catch (...) {
      byName.erase(owner);
      throw;
    }
  }
  void erase(hnswlib::labeltype label) {
    byName.erase(*byLabel.at(label));
    byLabel.erase(label);
  }
  size_t size() const { return byName.size(); }
  const std::unordered_map<hnswlib::labeltype, const std::string *> &
  entries() const {
    return byLabel;
  }

  void save(std::shared_ptr<OutputStream> stream) const {
    writeBinaryPOD(stream, uint64_t(size()));
    // Stable ordering makes repeated saves deterministic.
    std::vector<hnswlib::labeltype> labels;
    for (const auto &entry : byLabel)
      labels.push_back(entry.first);
    std::sort(labels.begin(), labels.end());
    for (auto label : labels) {
      const auto &value = name(label);
      writeBinaryPOD(stream, uint64_t(label));
      writeBinaryPOD(stream, uint64_t(value.size()));
      if (!stream->write(value.data(), value.size()))
        throw std::runtime_error("Failed to write Voyager string identifier.");
    }
  }
  static BiMap load(std::shared_ptr<InputStream> stream) {
    BiMap result;
    uint64_t count;
    readBinaryPOD(stream, count);
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t label, length;
      readBinaryPOD(stream, label);
      readBinaryPOD(stream, length);
      // Read incrementally: a corrupt length must not cause a huge allocation.
      std::string value;
      char buffer[4096];
      while (length) {
        auto chunk = std::min<uint64_t>(length, sizeof(buffer));
        if (stream->read(buffer, chunk) != static_cast<long long>(chunk))
          throw std::runtime_error(
              "Truncated Voyager string identifier mapping.");
        value.append(buffer, chunk);
        length -= chunk;
      }
      result.insert(label, value);
    }
    return result;
  }

private:
  // unordered_map preserves references to keys across rehashes. The owning
  // map is declared first so it is destroyed after the reverse references.
  std::unordered_map<std::string, hnswlib::labeltype> byName;
  std::unordered_map<hnswlib::labeltype, const std::string *> byLabel;
};
} // namespace voyager
