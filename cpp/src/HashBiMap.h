#include <iostream>
#include <stdexcept>
#include <unordered_map>

/**
 * @brief A bidirectional map that allows unique key-value pairs.
 *
 * This class provides a bidirectional map where each key is associated with a
 * unique value and each value is associated with a unique key. It supports
 * insertion, deletion, and lookup operations in both directions.
 *
 * This class does NOT allow key<->value mappings to be updated. Once a
 * key-value pair is inserted, it cannot be changed.
 *
 * @tparam K Type of the keys.
 * @tparam V Type of the values.
 */
template <typename K, typename V> class HashBiMap {
private:
  std::unordered_map<K, V> forwardMap;
  std::unordered_map<V, K> reverseMap;

public:
  // puts a key-value pair
  void put(const K &key, const V &value) {
    if (forwardMap.find(key) != forwardMap.end() ||
        reverseMap.find(value) != reverseMap.end()) {
      throw std::invalid_argument(
          "Duplicate key or value not allowed in HashBiMap");
    }
    forwardMap[key] = value;
    reverseMap[value] = key;
  }

  // Removes a key-value pair by key
  void removeByKey(const K &key) {
    auto it = forwardMap.find(key);
    if (it != forwardMap.end()) {
      V value = it->second;
      forwardMap.erase(it);
      reverseMap.erase(value);
    }
  }

  // Removes a key-value pair by value
  void removeByValue(const V &value) {
    auto it = reverseMap.find(value);
    if (it != reverseMap.end()) {
      K key = it->second;
      reverseMap.erase(it);
      forwardMap.erase(key);
    }
  }

  // Retrieves the value associated with a key
  V getByKey(const K &key) const {
    auto it = forwardMap.find(key);
    if (it == forwardMap.end()) {
      throw std::out_of_range("Key not found");
    }
    return it->second;
  }

  // Retrieves the key associated with a value
  K getByValue(const V &value) const {
    auto it = reverseMap.find(value);
    if (it == reverseMap.end()) {
      throw std::out_of_range("Value not found");
    }
    return it->second;
  }

  // Checks if the map contains a given key
  bool containsKey(const K &key) const {
    return forwardMap.find(key) != forwardMap.end();
  }

  // Checks if the map contains a given value
  bool containsValue(const V &value) const {
    return reverseMap.find(value) != reverseMap.end();
  }

  size_t getSize() const { return forwardMap.size(); }

  // Clears the entire map
  void clear() {
    forwardMap.clear();
    reverseMap.clear();
  }
};
