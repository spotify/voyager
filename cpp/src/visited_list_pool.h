/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
 *
 * This file is includes code from hnswlib (https://github.com/nmslib/hnswlib,
 * Apache 2.0-licensed, no copyright author listed)
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

#include <deque>
#include <mutex>
#include <string.h>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
public:
  vl_type curV;
  vl_type *mass;
  unsigned int numelements;

  VisitedList(int numelements1) {
    curV = -1;
    numelements = numelements1;
    mass = new vl_type[numelements];
  }

  void reset() {
    curV++;
    if (curV == 0) {
      memset(mass, 0, sizeof(vl_type) * numelements);
      curV++;
    }
  };

  ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
  std::deque<VisitedList *> pool;
  std::mutex poolguard;
  int numelements;

public:
  VisitedListPool(int initmaxpools, int numelements1) {
    numelements = numelements1;
    for (int i = 0; i < initmaxpools; i++)
      pool.push_front(new VisitedList(numelements));
  }

  VisitedList *getFreeVisitedList() {
    VisitedList *rez;
    {
      std::unique_lock<std::mutex> lock(poolguard);
      if (pool.size() > 0) {
        rez = pool.front();
        pool.pop_front();
      } else {
        rez = new VisitedList(numelements);
      }
    }
    rez->reset();
    return rez;
  };

  void releaseVisitedList(VisitedList *vl) {
    std::unique_lock<std::mutex> lock(poolguard);
    pool.push_front(vl);
  };

  ~VisitedListPool() {
    while (pool.size()) {
      VisitedList *rez = pool.front();
      pool.pop_front();
      delete rez;
    }
  };
};
} // namespace hnswlib
