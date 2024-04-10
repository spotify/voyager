package com.spotify.voyager.utils;

import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class ConcurrentBiMap<K, V> implements Map<K, V>  {
  private final Map<K, V> forwardMap;
  private final Map<V, K> reverseMap;

  public ConcurrentBiMap() {
    this(new ConcurrentHashMap<>(16, 0.75f, 1), new ConcurrentHashMap<>(16, 0.75f, 1));
  }

  public ConcurrentBiMap(int initialCapacity) {
    this(new ConcurrentHashMap<>(initialCapacity, 0.75f, 1), new ConcurrentHashMap<>(initialCapacity, 0.75f, 1));
  }

  private ConcurrentBiMap(Map<K, V> forwardMap, Map<V, K> reverseMap) {
    this.forwardMap = forwardMap;
    this.reverseMap = reverseMap;
  }
  @Override
  public int size() {
    return forwardMap.size();
  }

  @Override
  public boolean isEmpty() {
    return forwardMap.isEmpty();
  }

  @Override
  public boolean containsKey(Object key) {
    return forwardMap.containsKey(key);
  }

  @Override
  public boolean containsValue(Object value) {
    return reverseMap.containsKey(value);
  }

  @Override
  public V get(Object key) {
    return forwardMap.get(key);
  }

  public K getKey(V value) {
    return reverseMap.get(value);
  }

  /**
   * Associates the specified key with the specified value, throwing an IllegalArgumentException
   * if the value is already associated with a different key.
   *
   * @param key key with which the specified value is to be associated
   * @param value value to be associated with the specified key
   * @return previous value associated with the passed key
   */
  @Override
  public synchronized V put(K key, V value) {
    K oldKey = reverseMap.get(value);
    if (oldKey != null && !key.equals(oldKey))
      throw new IllegalArgumentException(value + " is already bound in reverseMap to " + oldKey);
    V oldVal = forwardMap.put(key, value);
    if (oldVal != null && !Objects.equals(reverseMap.remove(oldVal), key))
      throw new IllegalStateException("Bad reverse mapping: " + oldVal + " and " + reverseMap.remove(oldVal));
    reverseMap.put(value, key);
    return oldVal;
  }

  /**
   * Associate a specified key with a specified value, ignoring any previous state. Will not throw
   * an exception if the value was already mapped from another key.
   * @param key key with which the specified value is to be associated
   * @param value value to be associated with the specified key
   * @return the value which was previously associated with the key, which may be null,
   * or null if there was no previous entry
   */
  public synchronized V forcePut(K key, V value) {
    V oldVal = forwardMap.put(key, value);
    reverseMap.put(value, key);
    return oldVal;
  }

  @Override
  public synchronized V remove(Object key) {
    V oldVal = forwardMap.remove(key);
    if (oldVal == null)
      return null;
    Object oldKey = reverseMap.remove(oldVal);
    if (oldKey == null || !oldKey.equals(key))
      throw new IllegalStateException("Invalid reverse mapping: " + key + " and " + oldKey);
    return oldVal;
  }

  @Override
  public synchronized void putAll(Map<? extends K, ? extends V> m) {
    for (Entry<? extends K, ? extends V> entry : m.entrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }

  @Override
  public void clear() {
    forwardMap.clear();
    reverseMap.clear();
  }

  @Override
  public Set<K> keySet() {
    return forwardMap.keySet();
  }

  @Override
  public Collection<V> values() {
    return reverseMap.keySet();
  }

  @Override
  public Set<Entry<K, V>> entrySet() {
    return forwardMap.entrySet();
  }
}
