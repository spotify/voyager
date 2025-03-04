#pragma once

#include <functional>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <vector>

/**
 * @class HashBiMap
 * @brief A bidirectional hash map implementation that allows for efficient
 *        key-to-value and value-to-key lookups. This implementation is inspired
 *        by Guava's HashBiMap in Java. Includes serialization and
 * deserialization methods.
 *
 * @tparam K The type of keys maintained by this map.
 * @tparam V The type of mapped values.
 *
 * @note This implementation is not thread-safe.
 */
template <typename K, typename V> class HashBiMap {
public:
  /**
   * @brief Constructs an empty HashBiMap.
   */
  HashBiMap() : size(0), mask(15) { init(16); }

  /**
   * @brief Destroys the HashBiMap and frees all associated memory.
   */
  ~HashBiMap() { clear(); }

  /**
   * @brief Inserts a key-value pair into the map.
   *
   * @throws std::invalid_argument if the value is already present and force is
   * false.
   */
  void put(const K &key, const V &value) { put(key, value, false); }

  /**
   * @brief Inserts a key-value pair into the map, overwriting any existing
   * entry with the same value.
   */
  void forcePut(const K &key, const V &value) { put(key, value, true); }

  /**
   * @brief Retrieves the value associated with the given key.
   *
   * @throws std::out_of_range if the key is not found.
   */
  V get(const K &key) const {
    auto entry = seekByKey(key, smearedHash(key));
    if (entry == nullptr) {
      throw std::out_of_range("Key not found");
    }
    return entry->value;
  }

  /**
   * @brief Retrieves the key associated with the given value.
   *
   * @throws std::out_of_range if the value is not found.
   */
  K getInverse(const V &value) const {
    auto entry = seekByValue(value, smearedHash(value));
    if (entry == nullptr) {
      throw std::out_of_range("Value not found");
    }
    return entry->key;
  }

  /**
   * @brief Removes the entry with the given key.
   */
  void remove(const K &key) {
    auto entry = seekByKey(key, smearedHash(key));
    if (entry != nullptr) {
      deleteEntry(entry);
    }
  }

  /**
   * @brief Removes the entry with the given value.
   */
  void removeInverse(const V &value) {
    auto entry = seekByValue(value, smearedHash(value));
    if (entry != nullptr) {
      deleteEntry(entry);
    }
  }

  /**
   * @brief Checks if the map contains the given key.
   *
   * @return True if the key is present, false otherwise.
   */
  bool containsKey(const K &key) const {
    return seekByKey(key, smearedHash(key)) != nullptr;
  }

  /**
   * @brief Checks if the map contains the given value.
   *
   * @return True if the value is present, false otherwise.
   */
  bool containsValue(const V &value) const {
    return seekByValue(value, smearedHash(value)) != nullptr;
  }

  /**
   * @brief Clears all entries from the map.
   */
  void clear() {

    BiEntry *entry = firstInKeyInsertionOrder;
    while (entry != nullptr) {
      BiEntry *nextEntry = entry->nextInKeyInsertionOrder;
      delete entry;
      entry = nextEntry;
    }

    size = 0;
    std::fill(hashTableKToV.begin(), hashTableKToV.end(), nullptr);
    std::fill(hashTableVToK.begin(), hashTableVToK.end(), nullptr);
    firstInKeyInsertionOrder = nullptr;
    lastInKeyInsertionOrder = nullptr;
  }

  /**
   * @brief Returns the number of entries in the map.
   */
  size_t getSize() const { return size; }

private:
  struct BiEntry {
    K key;
    V value;
    int keyHash;
    int valueHash;
    BiEntry *nextInKToVBucket;        ///< Pointer to the next entry in the
                                      ///< key-to-value bucket.
    BiEntry *nextInVToKBucket;        ///< Pointer to the next entry in the
                                      ///< value-to-key bucket.
    BiEntry *nextInKeyInsertionOrder; ///< Pointer to the next entry in key
                                      ///< insertion order.
    BiEntry *prevInKeyInsertionOrder; ///< Pointer to the previous entry in key
                                      ///< insertion order.

    /**
     * @brief Constructs a BiEntry with the given key, key hash, value, and
     * value hash.
     *
     * @param key The key of the entry.
     * @param keyHash The hash code of the key.
     * @param value The value of the entry.
     * @param valueHash The hash code of the value.
     */
    BiEntry(const K &key, int keyHash, const V &value, int valueHash)
        : key(key), value(value), keyHash(keyHash), valueHash(valueHash),
          nextInKToVBucket(nullptr), nextInVToKBucket(nullptr),
          nextInKeyInsertionOrder(nullptr), prevInKeyInsertionOrder(nullptr) {}
  };

  std::vector<BiEntry *> hashTableKToV;
  std::vector<BiEntry *> hashTableVToK;
  BiEntry *firstInKeyInsertionOrder;
  BiEntry *lastInKeyInsertionOrder;
  size_t size;
  int mask;

  /**
   * @brief Initializes the hash tables with the expected size.
   *
   * @param expectedSize The expected number of entries.
   */
  void init(int expectedSize) {
    int tableSize = closedTableSize(expectedSize, 1.0);
    hashTableKToV.resize(tableSize, nullptr);
    hashTableVToK.resize(tableSize, nullptr);
    firstInKeyInsertionOrder = nullptr;
    lastInKeyInsertionOrder = nullptr;
    size = 0;
    mask = tableSize - 1;
  }

  /**
   * @brief Computes the hash code for the given key.
   *
   * @return The hash code for the key.
   */
  int smearedHash(const K &key) const { return std::hash<K>{}(key); }

  /**
   * @brief Computes the hash code for the given value.
   *
   * @return The hash code for the value.
   */
  int smearedHash(const V &value) const { return std::hash<V>{}(value); }

  /**
   * @brief Searches for an entry by key.
   *
   * @param key The key to search for.
   * @param keyHash The hash code of the key.
   * @return A pointer to the entry if found, nullptr otherwise.
   */
  BiEntry *seekByKey(const K &key, int keyHash) const {
    for (BiEntry *entry = hashTableKToV[keyHash & mask]; entry != nullptr;
         entry = entry->nextInKToVBucket) {
      if (keyHash == entry->keyHash && key == entry->key) {
        return entry;
      }
    }
    return nullptr;
  }

  /**
   * @brief Searches for an entry by value.
   *
   * @param value The value to search for.
   * @param valueHash The hash code of the value.
   * @return A pointer to the entry if found, nullptr otherwise.
   */
  BiEntry *seekByValue(const V &value, int valueHash) const {
    for (BiEntry *entry = hashTableVToK[valueHash & mask]; entry != nullptr;
         entry = entry->nextInVToKBucket) {
      if (valueHash == entry->valueHash && value == entry->value) {
        return entry;
      }
    }
    return nullptr;
  }

  /**
   * @brief Deletes an entry from the map.
   */
  void deleteEntry(BiEntry *entry) {
    int keyBucket = entry->keyHash & mask;
    BiEntry *prevBucketEntry = nullptr;
    for (BiEntry *bucketEntry = hashTableKToV[keyBucket];
         bucketEntry != nullptr; bucketEntry = bucketEntry->nextInKToVBucket) {
      if (bucketEntry == entry) {
        if (prevBucketEntry == nullptr) {
          hashTableKToV[keyBucket] = entry->nextInKToVBucket;
        } else {
          prevBucketEntry->nextInKToVBucket = entry->nextInKToVBucket;
        }
        break;
      }
      prevBucketEntry = bucketEntry;
    }

    int valueBucket = entry->valueHash & mask;
    prevBucketEntry = nullptr;
    for (BiEntry *bucketEntry = hashTableVToK[valueBucket];
         bucketEntry != nullptr; bucketEntry = bucketEntry->nextInVToKBucket) {
      if (bucketEntry == entry) {
        if (prevBucketEntry == nullptr) {
          hashTableVToK[valueBucket] = entry->nextInVToKBucket;
        } else {
          prevBucketEntry->nextInVToKBucket = entry->nextInVToKBucket;
        }
        break;
      }
      prevBucketEntry = bucketEntry;
    }

    if (entry->prevInKeyInsertionOrder == nullptr) {
      firstInKeyInsertionOrder = entry->nextInKeyInsertionOrder;
    } else {
      entry->prevInKeyInsertionOrder->nextInKeyInsertionOrder =
          entry->nextInKeyInsertionOrder;
    }

    if (entry->nextInKeyInsertionOrder == nullptr) {
      lastInKeyInsertionOrder = entry->prevInKeyInsertionOrder;
    } else {
      entry->nextInKeyInsertionOrder->prevInKeyInsertionOrder =
          entry->prevInKeyInsertionOrder;
    }

    delete entry;
    size--;
  }

  /**
   * @brief Inserts an entry into the map.
   *
   * @param entry The entry to insert.
   * @param oldEntryForKey The old entry for the key, if any.
   */
  void insert(BiEntry *entry, BiEntry *oldEntryForKey) {
    int keyBucket = entry->keyHash & mask;
    entry->nextInKToVBucket = hashTableKToV[keyBucket];
    hashTableKToV[keyBucket] = entry;

    int valueBucket = entry->valueHash & mask;
    entry->nextInVToKBucket = hashTableVToK[valueBucket];
    hashTableVToK[valueBucket] = entry;

    if (oldEntryForKey == nullptr) {
      entry->prevInKeyInsertionOrder = lastInKeyInsertionOrder;
      entry->nextInKeyInsertionOrder = nullptr;
      if (lastInKeyInsertionOrder == nullptr) {
        firstInKeyInsertionOrder = entry;
      } else {
        lastInKeyInsertionOrder->nextInKeyInsertionOrder = entry;
      }
      lastInKeyInsertionOrder = entry;
    } else {
      entry->prevInKeyInsertionOrder = oldEntryForKey->prevInKeyInsertionOrder;
      if (entry->prevInKeyInsertionOrder == nullptr) {
        firstInKeyInsertionOrder = entry;
      } else {
        entry->prevInKeyInsertionOrder->nextInKeyInsertionOrder = entry;
      }
      entry->nextInKeyInsertionOrder = oldEntryForKey->nextInKeyInsertionOrder;
      if (entry->nextInKeyInsertionOrder == nullptr) {
        lastInKeyInsertionOrder = entry;
      } else {
        entry->nextInKeyInsertionOrder->prevInKeyInsertionOrder = entry;
      }
    }

    size++;
  }

  /**
   * @brief Inserts a key-value pair into the map, with an option to force
   * insertion.
   *
   * @param key The key to insert.
   * @param value The value to insert.
   * @param force Whether to force insertion, overwriting any existing entry
   * with the same value.
   */
  void put(const K &key, const V &value, bool force) {
    int keyHash = smearedHash(key);
    int valueHash = smearedHash(value);

    BiEntry *oldEntryForKey = seekByKey(key, keyHash);
    if (oldEntryForKey != nullptr && valueHash == oldEntryForKey->valueHash &&
        value == oldEntryForKey->value) {
      return;
    }

    BiEntry *oldEntryForValue = seekByValue(value, valueHash);
    if (oldEntryForValue != nullptr) {
      if (force) {
        deleteEntry(oldEntryForValue);
      } else {
        throw std::invalid_argument("Value already present");
      }
    }

    BiEntry *newEntry = new BiEntry(key, keyHash, value, valueHash);
    if (oldEntryForKey != nullptr) {
      deleteEntry(oldEntryForKey);
      insert(newEntry, oldEntryForKey);
    } else {
      insert(newEntry, nullptr);
      rehashIfNecessary();
    }
  }

  /**
   * @brief Rehashes the map if necessary to maintain the load factor.
   */
  void rehashIfNecessary() {
    if (needsResizing(size, hashTableKToV.size(), 1.0)) {
      int newTableSize = hashTableKToV.size() * 2;
      hashTableKToV.resize(newTableSize, nullptr);
      hashTableVToK.resize(newTableSize, nullptr);
      mask = newTableSize - 1;
      size = 0;

      for (BiEntry *entry = firstInKeyInsertionOrder; entry != nullptr;
           entry = entry->nextInKeyInsertionOrder) {
        insert(entry, entry);
      }
    }
  }

  /**
   * @brief Determines if the map needs resizing based on the current size and
   * load factor.
   *
   * @param size The current size of the map.
   * @param tableSize The current size of the hash table.
   * @param loadFactor The load factor threshold.
   * @return True if resizing is needed, false otherwise.
   */
  bool needsResizing(size_t size, size_t tableSize, double loadFactor) const {
    return size > tableSize * loadFactor;
  }

  /**
   * @brief Computes the closed table size based on the expected size and load
   * factor.
   *
   * @param expectedSize The expected number of entries.
   * @param loadFactor The load factor threshold.
   * @return The computed table size.
   */
  int closedTableSize(int expectedSize, double loadFactor) const {
    int tableSize = 1;
    while (tableSize < expectedSize / loadFactor) {
      tableSize <<= 1;
    }
    return tableSize;
  }
};
