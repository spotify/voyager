#include "HashBiMap.h"
#include "doctest.h"

TEST_CASE("HashBiMap: put and getByKey") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  REQUIRE(map.getByKey("one") == 1);
  REQUIRE(map.getByKey("two") == 2);
  REQUIRE(map.getSize() == 2);

  REQUIRE_THROWS_AS(map.put("one", 1), std::invalid_argument);
  // throw exception if key is already present
  REQUIRE_THROWS_AS(map.put("one", 99), std::invalid_argument);
  // throw exception if value is already present
  REQUIRE_THROWS_AS(map.put("foo", 1), std::invalid_argument);
}

TEST_CASE("HashBiMap: put and getByValue") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  REQUIRE(map.getByValue(1) == "one");
  REQUIRE(map.getByValue(2) == "two");
}

TEST_CASE("HashBiMap: removeByKey") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  map.removeByKey("one");
  REQUIRE(!map.containsKey("one"));
  REQUIRE(!map.containsValue(1));
  REQUIRE(map.getSize() == 1);
}

TEST_CASE("HashBiMap: removeByValue") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  map.removeByValue(1);
  REQUIRE(!map.containsKey("one"));
  REQUIRE(!map.containsValue(1));
  REQUIRE(map.getSize() == 1);
}

TEST_CASE("HashBiMap: clear") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  map.clear();
  REQUIRE(map.getSize() == 0);
  REQUIRE(!map.containsKey("one"));
  REQUIRE(!map.containsValue(2));
}

TEST_CASE("HashBiMap: containsKey and containsValue") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  REQUIRE(map.containsKey("one"));
  REQUIRE(map.containsValue(1));
  REQUIRE(map.containsKey("two"));
  REQUIRE(map.containsValue(2));
}
