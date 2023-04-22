#pragma once
#include <functional>

namespace hnswlib {
template <typename MTYPE, typename data_t = MTYPE>
using DISTFUNC =
    std::function<MTYPE(const data_t *, const data_t *, const size_t)>;

/**
 * An abstract class representing a type of space to search through,
 * and encapsulating the data required to search that space.
 *
 * Possible subclasses include Euclidean, InnerProduct, etc.
 */
template <typename MTYPE, typename data_t = MTYPE> class Space {
public:
  virtual size_t get_data_size() = 0;

  virtual DISTFUNC<MTYPE, data_t> get_dist_func() = 0;

  virtual size_t get_dist_func_param() = 0;

  virtual ~Space() {}
};
}; // namespace hnswlib
