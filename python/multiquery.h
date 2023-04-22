template <typename dist_t, typename data_t = dist_t,
          typename scalefactor = std::ratio<1, 1>>
py::object knnQuery_multiple_return_numpy(
    std::vector<std::shared_ptr<Index<dist_t, data_t, scalefactor>>> indices,
    py::buffer input, size_t k, int num_threads, hnswlib::labeltype idMask,
    size_t minItemMatchesPerIndex, size_t k_per_index, dist_t maximumDistance) {

  if (indices.empty()) {
    throw std::domain_error("At least one index must be passed!");
  }

  if (k_per_index == 0) {
    k_per_index = k;
  }

  bool normalize = false;
  size_t dimensions = 0;
  size_t totalElementCount = 0;
  for (int i = 0; i < indices.size(); i++) {
    if (!indices[i]->index_inited) {
      throw std::runtime_error(
          "Can't make queries before all indices are initialized!");
    }

    // All indices must be homogenous here:
    if (i == 0) {
      normalize = indices[i]->normalize;
    } else if (indices[i]->normalize != normalize) {
      throw std::runtime_error(
          "All indices must share the same distance metric.");
    }

    if (i == 0) {
      dimensions = indices[i]->dim;
    } else if (indices[i]->dim != dimensions) {
      throw std::runtime_error(
          "All indices must share the same number of dimensions.");
    }

    totalElementCount += indices[i]->getCurrentCount();
  }

  if (totalElementCount < k) {
    throw std::runtime_error(
        "Sum of all elements across all indices is not enough to return "
        "requested number of neighbors. (Asked for " +
        std::to_string(k) + " neighbors, while only " +
        std::to_string(totalElementCount) +
        " vectors are available across all " + std::to_string(indices.size()) +
        " indices.)");
  }

  py::buffer_info info = input.request();
  py::array_t<data_t, py::array::c_style | py::array::forcecast> items(input);
  auto buffer = items.request();

  hnswlib::labeltype *data_numpy_labels;
  unsigned short *data_numpy_index_ids;
  dist_t *data_numpy_distances;
  size_t rows, features;

  if (num_threads <= 0)
    num_threads = std::thread::hardware_concurrency();

  {
    py::gil_scoped_release l;

    if (buffer.ndim != 2 && buffer.ndim != 1)
      throw std::runtime_error("data must be a 1d/2d array");
    if (buffer.ndim == 2) {
      rows = buffer.shape[0];
      features = buffer.shape[1];
    } else {
      rows = 1;
      features = buffer.shape[0];
    }

    // Handle rescaling to integer storage values if necessary:
    if constexpr (scalefactor::num != scalefactor::den) {
      if (info.format == py::format_descriptor<data_t>::format()) {
        // Don't do anything; assume that the inputs are already the correct
        // value and will not be scaled.
      } else if (info.format == py::format_descriptor<float>::format()) {
        // Re-scale the input values by multiplying by `scalefactor`:
        constexpr float lowerBound = (float)std::numeric_limits<data_t>::min() *
                                     (float)scalefactor::num /
                                     (float)scalefactor::den;
        constexpr float upperBound = (float)std::numeric_limits<data_t>::max() *
                                     (float)scalefactor::num /
                                     (float)scalefactor::den;

        py::array_t<float, py::array::c_style> floatItems(input);
        // Re-scale the input values by multiplying by `scalefactor`:
        auto vector = items.template mutable_unchecked<2>();
        for (size_t i = 0; i < rows; i++) {
          const float *inputVector = floatItems.data(i);
          for (int j = 0; j < features; j++) {
            if (inputVector[j] > upperBound || inputVector[j] < lowerBound) {
              throw std::domain_error(
                  "One or more vectors contain values outside of [" +
                  std::to_string(lowerBound) + ", " +
                  std::to_string(upperBound) + "].");
            }
            vector(i, j) = (inputVector[j] * (float)scalefactor::den) /
                           (float)scalefactor::num;
          }
        }
      } else {
        throw std::domain_error("Items added to index must be " +
                                typeName<data_t>() + " or float32!");
      }
    }

    data_numpy_labels = new hnswlib::labeltype[rows * k];
    data_numpy_index_ids = new unsigned short[rows * k];
    data_numpy_distances = new dist_t[rows * k];

    // Allocate one VisitedList (query-time data structure) to be used
    // per query thread. Each list should be big enough for any of the indices.
    size_t maxIndexElementCount = 0;
    for (int i = 0; i < indices.size(); i++) {
      maxIndexElementCount =
          std::max(indices[i]->getCurrentCount(), maxIndexElementCount);
    }

    // VisitedList's move constructor breaks if we put it into a vector, so we
    // use unique_ptr:
    std::vector<std::unique_ptr<hnswlib::VisitedList>> visitedLists;
    for (int i = 0; i < num_threads; i++) {
      visitedLists.push_back(
          std::make_unique<hnswlib::VisitedList>(maxIndexElementCount));
    }

    std::vector<
        std::pair<std::mutex, std::priority_queue<std::tuple<
                                  dist_t, unsigned short, hnswlib::labeltype>>>>
        resultsPerVector(rows);

    std::vector<data_t> norm_array(num_threads * features);
    ParallelFor(
        0, indices.size(), num_threads, [&](size_t indexId, size_t threadId) {
          std::vector<
              std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>>
              shardResults(rows);

          // Group by ID and only consider IDs that appear in
          // at least minItemMatchesPerIndex query results:
          std::map<hnswlib::labeltype, std::set<size_t>> rowIndicesPerID;

          for (size_t row = 0; row < rows; row++) {
            data_t *vectorToSearchWith = (data_t *)items.data(row);

            if (normalize) {
              size_t start_idx = threadId * dimensions;
              indices[indexId]->normalize_vector(
                  vectorToSearchWith, (norm_array.data() + start_idx));
              vectorToSearchWith = (norm_array.data() + start_idx);
            }

            shardResults[row] = indices[indexId]->appr_alg->searchKnn(
                vectorToSearchWith, k_per_index, visitedLists[threadId].get());

            if (minItemMatchesPerIndex > 1) {
              std::vector<std::pair<dist_t, hnswlib::labeltype>> &items =
                  GetContainerForQueue(shardResults[row]);
              for (auto i = items.begin(); i != items.end(); i++) {
                rowIndicesPerID[i->second & idMask].insert(row);
              }
            }
          }

          // Only keep IDs that show up in at least minItemMatchesPerIndex
          // separate queries:
          std::set<hnswlib::labeltype> idsToKeep;
          if (minItemMatchesPerIndex > 1) {
            for (auto idAndCount : rowIndicesPerID) {
              if (idAndCount.second.size() >= minItemMatchesPerIndex) {
                idsToKeep.insert(idAndCount.first);
              }
            }
          }

          // Merge the results into the output queue(s):
          for (size_t i = 0; i < rows; i++) {
            // Avoid lock contention by starting at a different
            // place in the output array in each thread:
            size_t row = (i + ((threadId * rows) / num_threads)) % rows;
            std::unique_lock<std::mutex> lock(resultsPerVector[row].first);
            mergePriorityQueues(resultsPerVector[row].second, shardResults[row],
                                k, (unsigned short)indexId, idMask, idsToKeep,
                                maximumDistance);
          }
        });

    ParallelFor(0, rows, (rows >= num_threads) ? num_threads : 1,
                [&](size_t row, size_t threadId) {
                  auto result = resultsPerVector[row].second;

                  // If we don't have enough results to return, return
                  // a distance of 1,000,000, an index ID of 0, and a label of
                  // 0.
                  if (result.size() < k) {
                    for (int i = k - 1; i > (int)result.size() - 1; i--) {
                      data_numpy_distances[row * k + i] = 1000000;
                      data_numpy_index_ids[row * k + i] = 0;
                      data_numpy_labels[row * k + i] = 0;
                    }
                  }

                  for (int i = result.size() - 1; i >= 0; i--) {
                    auto &result_tuple = result.top();
                    data_numpy_distances[row * k + i] =
                        std::get<0>(result_tuple);
                    data_numpy_index_ids[row * k + i] =
                        std::get<1>(result_tuple);
                    data_numpy_labels[row * k + i] = std::get<2>(result_tuple);
                    result.pop();
                  }
                });
  }
  py::capsule free_when_done_l(data_numpy_labels, [](void *f) { delete[] f; });
  py::capsule free_when_done_i(data_numpy_index_ids,
                               [](void *f) { delete[] f; });
  py::capsule free_when_done_d(data_numpy_distances,
                               [](void *f) { delete[] f; });

  return py::make_tuple(
      py::array_t<hnswlib::labeltype>(
          {rows, k}, // shape
          {k * sizeof(hnswlib::labeltype),
           sizeof(hnswlib::labeltype)}, // C-style contiguous strides for
                                        // labeltype
          data_numpy_labels,            // the data pointer
          free_when_done_l),
      py::array_t<unsigned short>(
          {rows, k}, // shape
          {k * sizeof(unsigned short),
           sizeof(unsigned short)}, // C-style contiguous strides for
                                    // labeltype
          data_numpy_index_ids,     // the data pointer
          free_when_done_i),
      py::array_t<dist_t>(
          {rows, k}, // shape
          {k * sizeof(dist_t),
           sizeof(dist_t)},     // C-style contiguous strides for dist_t
          data_numpy_distances, // the data pointer
          free_when_done_d));
}