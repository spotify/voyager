/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2026 Spotify AB
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

#include <cstddef>
#include <functional>
#include <optional>
#include <string>

namespace voyager {
namespace cpu {
namespace detail {

using FileReader =
    std::function<std::optional<std::string>(const std::string &)>;

std::optional<std::size_t> parseCgroupV2CpuMax(const std::string &cpuMax);
std::optional<std::size_t>
parseCgroupV1CpuLimit(const std::string &quotaValue,
                      const std::string &periodValue);
std::optional<std::size_t> cgroupCpuCount(const FileReader &reader);
std::size_t
availableCpuCountFromLimits(std::size_t hardwareCount,
                            std::optional<std::size_t> affinityCount,
                            std::optional<std::size_t> quotaCount);

} // namespace detail
} // namespace cpu
} // namespace voyager
