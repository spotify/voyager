/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 *
 * This file is includes code from hnswlib (https://github.com/nmslib/hnswlib,
 * Apache 2.0-licensed, no copyright author listed)
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
#include <functional>

namespace hnswlib {
template <typename MTYPE, typename data_t = MTYPE>
using DISTFUNC =
    std::function<MTYPE(const data_t *, const data_t *, const size_t)>;

/**
 * An abstract class representing a type of space to search through,
 * and encapsulating the data required to search that space.
 *
 * Possible subclasses include Euclidean, InnerProduct, etc.
 */
template <typename MTYPE, typename data_t = MTYPE> class Space {
public:
  virtual size_t get_data_size() = 0;

  virtual DISTFUNC<MTYPE, data_t> get_dist_func() = 0;

  virtual size_t get_dist_func_param() = 0;

  virtual ~Space() {}
};
}; // namespace hnswlib
