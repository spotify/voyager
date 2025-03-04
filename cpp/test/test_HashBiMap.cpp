#include "HashBiMap.h"
#include "doctest.h"

TEST_CASE("HashBiMap: put and get") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);
  // same state if the key-value pair already exists
  map.put("one", 1);

  REQUIRE(map.get("one") == 1);
  REQUIRE(map.get("two") == 2);
  REQUIRE(map.getSize() == 2);

  // overwrite the value of key "one"
  map.put("one", 99);
  REQUIRE(map.get("one") == 99);

  // throw exception if value is already present with a different key
  REQUIRE_THROWS(map.put("foo", 99));
}

TEST_CASE("HashBiMap: forcePut") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);

  // This should overwrite the value 1 with key "new-one"
  map.forcePut("new-one", 1);

  REQUIRE(!map.containsKey("one"));
  REQUIRE(map.get("new-one") == 1);
  REQUIRE(map.getInverse(1) == "new-one");
  REQUIRE(map.getSize() == 1);

  map.put("two", 2);
  // same state if the key-value pair already exists
  map.forcePut("two", 2);

  REQUIRE(map.get("two") == 2);
  REQUIRE(map.getInverse(2) == "two");
  REQUIRE(map.getSize() == 2);
}

TEST_CASE("HashBiMap: getInverse") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  REQUIRE(map.getInverse(1) == "one");
  REQUIRE(map.getInverse(2) == "two");
}

TEST_CASE("HashBiMap: remove") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  map.remove("one");
  REQUIRE(!map.containsKey("one"));
  REQUIRE(!map.containsValue(1));
  REQUIRE(map.getSize() == 1);
}

TEST_CASE("HashBiMap: removeInverse") {
  HashBiMap<std::string, int> map;
  map.put("one", 1);
  map.put("two", 2);

  map.removeInverse(1);
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
