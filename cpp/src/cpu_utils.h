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

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace voyager {
namespace cpu {

using FileReader =
    std::function<std::optional<std::string>(const std::string &)>;

inline std::optional<std::string> readFile(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    return std::nullopt;

  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

inline std::vector<std::string> split(const std::string &value,
                                      char delimiter) {
  std::vector<std::string> parts;
  std::istringstream input(value);
  std::string part;
  while (std::getline(input, part, delimiter))
    parts.push_back(part);
  return parts;
}

inline bool containsToken(const std::string &list, const std::string &token) {
  const auto tokens = split(list, ',');
  return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

inline std::optional<size_t> parsePositiveInteger(const std::string &value) {
  try {
    size_t parsedCharacters = 0;
    const size_t firstCharacter = value.find_first_not_of(" \t\n\r");
    if (firstCharacter == std::string::npos || value[firstCharacter] == '-')
      return std::nullopt;

    const unsigned long long parsed = std::stoull(value, &parsedCharacters);
    if (parsed == 0 || parsed > std::numeric_limits<size_t>::max())
      return std::nullopt;

    while (parsedCharacters < value.size() &&
           std::isspace(static_cast<unsigned char>(value[parsedCharacters])))
      ++parsedCharacters;

    if (parsedCharacters != value.size())
      return std::nullopt;
    return static_cast<size_t>(parsed);
  } catch (...) {
    return std::nullopt;
  }
}

inline std::optional<size_t> cpuCountForQuota(size_t quota, size_t period) {
  if (quota == 0 || period == 0)
    return std::nullopt;
  return quota / period + (quota % period != 0);
}

inline std::optional<size_t> parseCgroupV2CpuMax(const std::string &cpuMax) {
  std::istringstream input(cpuMax);
  std::string quotaValue;
  std::string periodValue;
  input >> quotaValue >> periodValue;
  if (quotaValue.empty() || quotaValue == "max")
    return std::nullopt;

  const auto quota = parsePositiveInteger(quotaValue);
  const auto period = parsePositiveInteger(periodValue);
  if (!quota || !period)
    return std::nullopt;
  return cpuCountForQuota(*quota, *period);
}

inline std::optional<size_t>
parseCgroupV1CpuLimit(const std::string &quotaValue,
                      const std::string &periodValue) {
  if (!quotaValue.empty() && quotaValue[0] == '-')
    return std::nullopt;

  const auto quota = parsePositiveInteger(quotaValue);
  const auto period = parsePositiveInteger(periodValue);
  if (!quota || !period)
    return std::nullopt;
  return cpuCountForQuota(*quota, *period);
}

inline bool isOctalDigit(char value) { return value >= '0' && value <= '7'; }

inline bool hasOctalEscapeAt(const std::string &path, size_t offset) {
  return path[offset] == '\\' && offset + 3 < path.size() &&
         isOctalDigit(path[offset + 1]) && isOctalDigit(path[offset + 2]) &&
         isOctalDigit(path[offset + 3]);
}

inline std::string decodeMountInfoPath(const std::string &path) {
  std::string decoded;
  decoded.reserve(path.size());
  for (size_t i = 0; i < path.size(); ++i) {
    if (hasOctalEscapeAt(path, i)) {
      const char value =
          static_cast<char>((path[i + 1] - '0') * 64 + (path[i + 2] - '0') * 8 +
                            path[i + 3] - '0');
      decoded.push_back(value);
      i += 3;
    } else {
      decoded.push_back(path[i]);
    }
  }
  return decoded;
}

inline bool pathStartsWith(const std::string &path, const std::string &root) {
  if (root == "/")
    return !path.empty() && path[0] == '/';
  return path == root ||
         (path.size() > root.size() &&
          path.compare(0, root.size(), root) == 0 && path[root.size()] == '/');
}

struct CgroupMount {
  std::string root;
  std::string mountPoint;
  bool unified;
};

inline bool isRequestedCgroupMount(const std::vector<std::string> &fields,
                                   size_t separator, bool unified) {
  const std::string &fileSystem = fields[separator + 1];
  if (unified)
    return fileSystem == "cgroup2";
  return fileSystem == "cgroup" &&
         (containsToken(fields[separator + 3], "cpu") ||
          containsToken(fields[5], "cpu"));
}

inline std::optional<CgroupMount> parseCgroupMount(const std::string &line,
                                                   bool unified) {
  const auto fields = split(line, ' ');
  const auto separator = std::find(fields.begin(), fields.end(), "-");
  if (fields.size() < 6 || separator == fields.end())
    return std::nullopt;

  const size_t separatorIndex = separator - fields.begin();
  if (separatorIndex + 3 >= fields.size() ||
      !isRequestedCgroupMount(fields, separatorIndex, unified))
    return std::nullopt;

  return CgroupMount{decodeMountInfoPath(fields[3]),
                     decodeMountInfoPath(fields[4]), unified};
}

inline std::optional<CgroupMount>
findCpuCgroupMount(const std::string &mountInfo, const std::string &cgroupPath,
                   bool unified) {
  std::istringstream lines(mountInfo);
  std::optional<CgroupMount> bestMatch;
  std::string line;
  while (std::getline(lines, line)) {
    const auto mount = parseCgroupMount(line, unified);
    if (mount && pathStartsWith(cgroupPath, mount->root) &&
        (!bestMatch || mount->root.size() > bestMatch->root.size()))
      bestMatch = mount;
  }
  return bestMatch;
}

inline std::optional<std::string>
findCpuCgroupPath(const std::string &cgroupContents, bool unified) {
  std::istringstream lines(cgroupContents);
  std::string line;
  while (std::getline(lines, line)) {
    const size_t firstColon = line.find(':');
    const size_t secondColon = line.find(':', firstColon + 1);
    if (firstColon == std::string::npos || secondColon == std::string::npos)
      continue;

    const std::string controllers =
        line.substr(firstColon + 1, secondColon - firstColon - 1);
    if ((unified && controllers.empty()) ||
        (!unified && containsToken(controllers, "cpu")))
      return line.substr(secondColon + 1);
  }
  return std::nullopt;
}

inline std::optional<std::string>
resolveCgroupDirectory(const CgroupMount &mount,
                       const std::string &cgroupPath) {
  if (!pathStartsWith(cgroupPath, mount.root))
    return std::nullopt;

  const std::string suffix =
      mount.root == "/" ? cgroupPath : cgroupPath.substr(mount.root.size());
  if (suffix.empty() || suffix == "/")
    return mount.mountPoint;
  return mount.mountPoint + suffix;
}

inline std::string parentPath(const std::string &path) {
  const size_t separator = path.find_last_of('/');
  if (separator == std::string::npos)
    return {};
  if (separator == 0)
    return "/";
  return path.substr(0, separator);
}

inline std::optional<size_t> quotaAt(const std::string &directory, bool unified,
                                     const FileReader &reader) {
  if (unified) {
    const auto cpuMax = reader(directory + "/cpu.max");
    return cpuMax ? parseCgroupV2CpuMax(*cpuMax) : std::nullopt;
  }

  const auto quota = reader(directory + "/cpu.cfs_quota_us");
  const auto period = reader(directory + "/cpu.cfs_period_us");
  if (!quota || !period)
    return std::nullopt;
  return parseCgroupV1CpuLimit(*quota, *period);
}

inline std::optional<size_t> tightestQuota(const std::string &directory,
                                           const CgroupMount &mount,
                                           const FileReader &reader) {
  std::optional<size_t> result;
  std::string current = directory;
  while (pathStartsWith(current, mount.mountPoint)) {
    const auto limit = quotaAt(current, mount.unified, reader);
    if (limit)
      result = result ? std::min(*result, *limit) : limit;
    if (current == mount.mountPoint)
      break;
    current = parentPath(current);
  }
  return result;
}

inline std::optional<size_t> cgroupCpuCount(const FileReader &reader) {
  const auto cgroup = reader("/proc/self/cgroup");
  const auto mountInfo = reader("/proc/self/mountinfo");
  if (!cgroup || !mountInfo)
    return std::nullopt;

  for (const bool unified : {true, false}) {
    const auto path = findCpuCgroupPath(*cgroup, unified);
    if (!path)
      continue;

    const auto mount = findCpuCgroupMount(*mountInfo, *path, unified);
    if (!mount)
      continue;

    const auto directory = resolveCgroupDirectory(*mount, *path);
    if (directory) {
      const auto quota = tightestQuota(*directory, *mount, reader);
      if (quota)
        return quota;
    }
  }
  return std::nullopt;
}

inline std::optional<size_t> affinityCpuCount() {
#ifdef __linux__
  cpu_set_t allowedCpus;
  CPU_ZERO(&allowedCpus);
  if (sched_getaffinity(0, sizeof(allowedCpus), &allowedCpus) != 0)
    return std::nullopt;

  size_t count = 0;
  for (size_t cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    count += CPU_ISSET(cpu, &allowedCpus) != 0;
  return count == 0 ? std::nullopt : std::optional<size_t>(count);
#else
  return std::nullopt;
#endif
}

inline size_t availableCpuCountFromLimits(size_t hardwareCount,
                                          std::optional<size_t> affinityCount,
                                          std::optional<size_t> quotaCount) {
  size_t available = std::max<size_t>(1, hardwareCount);
  if (affinityCount)
    available = std::min(available, *affinityCount);
  if (quotaCount)
    available = std::min(available, *quotaCount);
  return std::max<size_t>(1, available);
}

inline size_t
availableCpuCount(size_t hardwareCount = std::thread::hardware_concurrency(),
                  std::optional<size_t> affinityCount = affinityCpuCount(),
                  const FileReader &reader = readFile) {
#ifdef __linux__
  const auto quotaCount = cgroupCpuCount(reader);
#else
  (void)reader;
  const std::optional<size_t> quotaCount = std::nullopt;
#endif
  return availableCpuCountFromLimits(hardwareCount, affinityCount, quotaCount);
}

} // namespace cpu
} // namespace voyager
