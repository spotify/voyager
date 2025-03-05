#include <fstream>
#include <iostream>
#include <sstream>
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

  /**
   * @brief Saves the HashBiMap to a file.
   *
   * The format of this file will be "['key1','key2','key3',...]" where the
   * index position of the key is the value of the key in the map. For example
   * if the HashBiMap is {"foo" -> 2, "baz" -> 0, "bar" -> 1}, the file will
   * contain "['baz','bar','foo']".
   *
   * Therefore the assumptions are that
   *  - the map values are contiguous integers from [0,size_of_index]
   *
   * @param bimap The HashBiMap to save.
   * @param filename The name of the file to save the map to.
   */
  void saveNamesMappingToFile(const std::string &filename) {
    std::vector<std::string> keys(getSize());
    for (const auto &pair : forwardMap) {
      keys[pair.second] = pair.first;
    }

    std::ofstream outFile(filename);
    if (!outFile) {
      throw std::runtime_error("Unable to open file for writing");
    }

    outFile << "[";
    for (size_t i = 0; i < keys.size(); ++i) {
      outFile << "'" << keys[i] << "'";
      if (i < keys.size() - 1) {
        outFile << ",";
      }
    }
    outFile << "]";
    outFile.close();
  }

  /**
   * @brief Loads the HashBiMap from a file.
   *
   * The format of this file should be "['key1','key2','key3',...]" where the
   * index position of the key is the value of the key in the map.
   *
   * @param filename The name of the file to load the map from.
   * @return The loaded HashBiMap.
   */
  static HashBiMap<std::string, int>
  loadNamesMappingFromFile(const std::string &filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
      throw std::runtime_error("Unable to open file for reading");
    }

    std::string content;
    std::getline(inFile, content);
    inFile.close();

    if (content.front() != '[' || content.back() != ']') {
      throw std::runtime_error("Invalid file format");
    }

    content =
        content.substr(1, content.size() - 2); // Remove the square brackets

    std::vector<std::string> keys;
    std::stringstream ss(content);
    std::string item;
    while (std::getline(ss, item, ',')) {
      item.erase(std::remove(item.begin(), item.end(), '\''), item.end());
      keys.push_back(item);
    }

    HashBiMap<std::string, int> bimap;
    for (size_t i = 0; i < keys.size(); ++i) {
      bimap.put(keys[i], i);
    }

    return bimap;
  }
};
