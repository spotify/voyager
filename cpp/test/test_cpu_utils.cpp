#include "doctest.h"

#include "cpu_utils.h"
#include <unordered_map>

using voyager::cpu::FileReader;

namespace {

FileReader
readerFor(const std::unordered_map<std::string, std::string> &files) {
  return [&files](const std::string &path) -> std::optional<std::string> {
    const auto value = files.find(path);
    if (value == files.end())
      return std::nullopt;
    return value->second;
  };
}

} // namespace

TEST_CASE("cgroup v2 CPU quotas round fractional CPUs up") {
  CHECK(voyager::cpu::parseCgroupV2CpuMax("100000 100000") == 1);
  CHECK(voyager::cpu::parseCgroupV2CpuMax("150000 100000") == 2);
  CHECK(voyager::cpu::parseCgroupV2CpuMax("250000 100000") == 3);
  CHECK_FALSE(voyager::cpu::parseCgroupV2CpuMax("max 100000"));
}

TEST_CASE("cgroup v1 unlimited and malformed quotas are ignored") {
  CHECK(voyager::cpu::parseCgroupV1CpuLimit("200000", "100000") == 2);
  CHECK_FALSE(voyager::cpu::parseCgroupV1CpuLimit("-1", "100000"));
  CHECK_FALSE(voyager::cpu::parseCgroupV1CpuLimit("invalid", "100000"));
  CHECK_FALSE(voyager::cpu::parseCgroupV1CpuLimit("100000", "0"));
}

TEST_CASE("cgroup v2 uses the tightest quota in its hierarchy") {
  const std::unordered_map<std::string, std::string> files = {
      {"/proc/self/cgroup", "0::/services/voyager\n"},
      {"/proc/self/mountinfo",
       "36 25 0:32 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - "
       "cgroup2 cgroup rw\n"},
      {"/sys/fs/cgroup/cpu.max", "max 100000\n"},
      {"/sys/fs/cgroup/services/cpu.max", "200000 100000\n"},
      {"/sys/fs/cgroup/services/voyager/cpu.max", "400000 100000\n"},
  };

  CHECK(voyager::cpu::cgroupCpuCount(readerFor(files)) == 2);
}

TEST_CASE("cgroup v1 CPU controller mount and quota are detected") {
  const std::unordered_map<std::string, std::string> files = {
      {"/proc/self/cgroup", "5:memory:/service\n4:cpu,cpuacct:/service\n"},
      {"/proc/self/mountinfo",
       "30 25 0:27 / /sys/fs/cgroup/cpu rw,relatime - cgroup cgroup "
       "rw,cpu,cpuacct\n"},
      {"/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "-1\n"},
      {"/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000\n"},
      {"/sys/fs/cgroup/cpu/service/cpu.cfs_quota_us", "150000\n"},
      {"/sys/fs/cgroup/cpu/service/cpu.cfs_period_us", "100000\n"},
  };

  CHECK(voyager::cpu::cgroupCpuCount(readerFor(files)) == 2);
}

TEST_CASE("cgroup namespaces resolve relative to the mount root") {
  const std::unordered_map<std::string, std::string> files = {
      {"/proc/self/cgroup", "0::/container/child\n"},
      {"/proc/self/mountinfo",
       "35 25 0:31 /other /other/cgroup rw,relatime - cgroup2 cgroup rw\n"
       "36 25 0:32 /container /sys/fs/cgroup rw,relatime - cgroup2 "
       "cgroup rw\n"},
      {"/sys/fs/cgroup/cpu.max", "300000 100000\n"},
      {"/sys/fs/cgroup/child/cpu.max", "max 100000\n"},
  };

  CHECK(voyager::cpu::cgroupCpuCount(readerFor(files)) == 3);
}

TEST_CASE("hybrid cgroups fall back to the v1 CPU controller") {
  const std::unordered_map<std::string, std::string> files = {
      {"/proc/self/cgroup", "0::/unified\n4:cpu:/legacy\n"},
      {"/proc/self/mountinfo",
       "30 25 0:27 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu\n"
       "36 25 0:32 / /sys/fs/cgroup/unified rw - cgroup2 cgroup rw\n"},
      {"/sys/fs/cgroup/cpu/legacy/cpu.cfs_quota_us", "100000\n"},
      {"/sys/fs/cgroup/cpu/legacy/cpu.cfs_period_us", "100000\n"},
  };

  CHECK(voyager::cpu::cgroupCpuCount(readerFor(files)) == 1);
}

TEST_CASE("missing CPU cgroup membership is ignored") {
  const std::unordered_map<std::string, std::string> files = {
      {"/proc/self/cgroup", "5:memory:/service\n"},
      {"/proc/self/mountinfo",
       "30 25 0:27 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu\n"},
  };

  CHECK_FALSE(voyager::cpu::cgroupCpuCount(readerFor(files)));
}

TEST_CASE("available CPU count combines hardware, affinity, and quota") {
  CHECK(voyager::cpu::availableCpuCountFromLimits(16, 8, 2) == 2);
  CHECK(voyager::cpu::availableCpuCountFromLimits(16, 4, std::nullopt) == 4);
  CHECK(voyager::cpu::availableCpuCountFromLimits(2, 8, 6) == 2);
  CHECK(voyager::cpu::availableCpuCount(0, std::nullopt, readerFor({})) == 1);
}
