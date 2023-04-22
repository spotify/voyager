#pragma once

#include <assert.h>
#include <atomic>
#include <iostream>
#include <ratio>
#include <set>
#include <stdlib.h>
#include <thread>

/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn) {
  if (numThreads <= 0) {
    numThreads = std::thread::hardware_concurrency();
  }

  if (numThreads == 1) {
    for (size_t id = start; id < end; id++) {
      fn(id, 0);
    }
  } else {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);

    // keep track of exceptions in threads
    // https://stackoverflow.com/a/32428427/1713196
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
      threads.push_back(std::thread([&, threadId] {
        while (true) {
          size_t id = current.fetch_add(1);

          if ((id >= end)) {
            break;
          }

          try {
            fn(id, threadId);
          } catch (...) {
            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
            lastException = std::current_exception();
            /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
            current = end;
            break;
          }
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }
    if (lastException) {
      std::rethrow_exception(lastException);
    }
  }
}

/**
 * A quick hack to allow unsorted, linear iteration over a std::priority_queue.
 * This dramatically speeds up filtering of an std::priority_queue, as you no
 * longer need to modify the queue to iterate over it.
 */
template <class T, class S, class C>
S &GetContainerForQueue(std::priority_queue<T, S, C> &q) {
  struct HackedQueue : private std::priority_queue<T, S, C> {
    static S &Container(std::priority_queue<T, S, C> &q) {
      return q.*&HackedQueue::c;
    }
  };
  return HackedQueue::Container(q);
}

/**
 * Merge the contents of two priority queues together, draining `src`
 * into `dest`, and keeping at most `maxElements`.
 *
 * IndexID will be added as the second tuple value of each element of the queue.
 */
template <typename dist_t, typename indexID_t, typename label_t>
void mergePriorityQueues(
    std::priority_queue<std::tuple<dist_t, indexID_t, label_t>> &dest,
    std::priority_queue<std::pair<dist_t, label_t>> &src, size_t maxElements,
    indexID_t indexID, const label_t idMask, const std::set<label_t> &labels,
    const dist_t maximumDistance) {
  std::vector<std::pair<dist_t, hnswlib::labeltype>> &items =
      GetContainerForQueue(src);
  for (auto i = items.begin(); i != items.end(); i++) {
    // To avoid copying unnecessarily, only move elements if:
    // - We don't have maxElements in `dest` yet
    // - `dest` is full, but the element being added would not be the new `top`
    // (i.e.: it's smaller)

    if (dest.size() < maxElements || i->first < std::get<0>(dest.top())) {
      if (idMask == 0 || labels.count(i->second & idMask) != 0) {
        if (i->first <= maximumDistance) {
          dest.push({i->first, indexID, i->second});
        }
      }
    }
  }

  while (dest.size() > maxElements)
    dest.pop();
}