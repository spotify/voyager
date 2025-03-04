#include "HashBiMap.h"
#include "doctest.h"

TEST_CASE("HashBiMap: put and get") {
  HashBiMap<int, std::string> map;
  map.put(1, "one");
  map.put(2, "two");

  REQUIRE(map.get(1) == "one");
  REQUIRE(map.get(2) == "two");
}

TEST_CASE("HashBiMap: forcePut") {
  HashBiMap<int, std::string> map;
  map.put(1, "one");
  map.forcePut(2, "one"); // This should overwrite the value "one" with key 2

  REQUIRE(map.get(2) == "one");
  REQUIRE(!map.containsKey(1));
}

TEST_CASE("HashBiMap: getInverse") {
  HashBiMap<int, std::string> map;
  map.put(1, "one");
  map.put(2, "two");

  REQUIRE(map.getInverse("one") == 1);
  REQUIRE(map.getInverse("two") == 2);
}

TEST_CASE("HashBiMap: remove") {
  HashBiMap<int, std::string> map;
  map.put(1, "one");
  map.put(2, "two");

  map.remove(1);
  REQUIRE(!map.containsKey(1));
  REQUIRE(!map.containsValue("one"));
  REQUIRE(map.getSize() == 1);
}

TEST_CASE("HashBiMap: removeInverse") {
  HashBiMap<int, std::string> map;
  map.put(1, "one");
  map.put(2, "two");

  map.removeInverse("one");
  REQUIRE(!map.containsKey(1));
  REQUIRE(!map.containsValue("one"));
  REQUIRE(map.getSize() == 1);
}

TEST_CASE("HashBiMap: clear") {
  HashBiMap<int, std::string> map;
  map.put(1, "one");
  map.put(2, "two");

  map.clear();
  REQUIRE(map.getSize() == 0);
  REQUIRE(!map.containsKey(1));
  REQUIRE(!map.containsValue("one"));
}

TEST_CASE("HashBiMap: containsKey and containsValue") {
  HashBiMap<int, std::string> map;
  map.put(1, "one");
  map.put(2, "two");

  REQUIRE(map.containsKey(1));
  REQUIRE(map.containsValue("one"));
  REQUIRE(map.containsKey(2));
  REQUIRE(map.containsValue("two"));
}
