#include <cstring>
#include <memory>
#include <napi.h>
#include <optional>
#include <sstream>
#include <vector>

// These are in either ../cpp/src (in dev mode) or ./voyager_src after prepack
// (during npm package installation)
#include "Enums.h"
#include "Index.h"
#include "Metadata.h"
#include "StreamUtils.h"
#include "TypedIndex.h"

// Local MemoryInputStream implementation for Node.js bindings
class MemoryInputStream : public InputStream {
public:
  MemoryInputStream(const std::string &data) : data(data), position(0) {}

  virtual bool isSeekable() { return true; }
  virtual long long getTotalLength() { return data.size(); }

  virtual long long read(char *buffer, long long bytesToRead) {
    long long bytesAvailable = data.size() - position;
    long long bytesToActuallyRead = std::min(bytesToRead, bytesAvailable);
    if (bytesToActuallyRead > 0) {
      std::memcpy(buffer, data.data() + position, bytesToActuallyRead);
      position += bytesToActuallyRead;
    }
    return bytesToActuallyRead;
  }

  virtual bool isExhausted() { return position >= data.size(); }
  virtual long long getPosition() { return position; }
  virtual bool setPosition(long long newPosition) {
    if (newPosition < 0 || newPosition > (long long)data.size()) {
      return false;
    }
    position = newPosition;
    return true;
  }

  virtual uint32_t peek() {
    if (position + sizeof(uint32_t) > data.size()) {
      throw std::runtime_error("Failed to peek " +
                               std::to_string(sizeof(uint32_t)) +
                               " bytes from memory stream at index " +
                               std::to_string(position) + ".");
    }
    uint32_t result = 0;
    std::memcpy(&result, data.data() + position, sizeof(uint32_t));
    return result;
  }

private:
  std::string data;
  long long position;
};

// Wrapper class for Index that works with Node-API
class IndexWrapper : public Napi::ObjectWrap<IndexWrapper> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);
  IndexWrapper(const Napi::CallbackInfo &info);

private:
  static Napi::FunctionReference constructor;

  std::shared_ptr<Index> index_;

  // Methods matching Python API
  Napi::Value AddItem(const Napi::CallbackInfo &info);
  Napi::Value AddItems(const Napi::CallbackInfo &info);
  Napi::Value Query(const Napi::CallbackInfo &info);
  Napi::Value GetVector(const Napi::CallbackInfo &info);
  Napi::Value GetVectors(const Napi::CallbackInfo &info);
  Napi::Value MarkDeleted(const Napi::CallbackInfo &info);
  Napi::Value UnmarkDeleted(const Napi::CallbackInfo &info);
  Napi::Value Resize(const Napi::CallbackInfo &info);
  Napi::Value SaveIndex(const Napi::CallbackInfo &info);
  static Napi::Value LoadIndex(const Napi::CallbackInfo &info);
  Napi::Value GetDistance(const Napi::CallbackInfo &info);

  // New methods for Buffer/Stream support
  Napi::Value ToBuffer(const Napi::CallbackInfo &info);
  static Napi::Value FromBuffer(const Napi::CallbackInfo &info);
  Napi::Value Has(const Napi::CallbackInfo &info);
  Napi::Value ToString(const Napi::CallbackInfo &info);

  // Property getters/setters
  Napi::Value GetSpace(const Napi::CallbackInfo &info);
  Napi::Value GetNumDimensions(const Napi::CallbackInfo &info);
  Napi::Value GetM(const Napi::CallbackInfo &info);
  Napi::Value GetEfConstruction(const Napi::CallbackInfo &info);
  Napi::Value GetMaxElements(const Napi::CallbackInfo &info);
  Napi::Value GetStorageDataType(const Napi::CallbackInfo &info);
  Napi::Value GetNumElements(const Napi::CallbackInfo &info);
  Napi::Value GetIds(const Napi::CallbackInfo &info);
  Napi::Value GetEf(const Napi::CallbackInfo &info);
  Napi::Value GetLength(const Napi::CallbackInfo &info);
  void SetEf(const Napi::CallbackInfo &info, const Napi::Value &value);
  void SetMaxElements(const Napi::CallbackInfo &info, const Napi::Value &value);

  // Helper methods
  std::vector<float> ArrayToVector(const Napi::Array &arr);
  Napi::Array VectorToArray(Napi::Env env, const std::vector<float> &vec);
};

Napi::FunctionReference IndexWrapper::constructor;

Napi::Object IndexWrapper::Init(Napi::Env env, Napi::Object exports) {
  Napi::HandleScope scope(env);

  Napi::Function func = DefineClass(
      env, "Index",
      {InstanceMethod("addItem", &IndexWrapper::AddItem),
       InstanceMethod("addItems", &IndexWrapper::AddItems),
       InstanceMethod("query", &IndexWrapper::Query),
       InstanceMethod("getVector", &IndexWrapper::GetVector),
       InstanceMethod("getVectors", &IndexWrapper::GetVectors),
       InstanceMethod("markDeleted", &IndexWrapper::MarkDeleted),
       InstanceMethod("unmarkDeleted", &IndexWrapper::UnmarkDeleted),
       InstanceMethod("resize", &IndexWrapper::Resize),
       InstanceMethod("saveIndex", &IndexWrapper::SaveIndex),
       StaticMethod("loadIndex", &IndexWrapper::LoadIndex),
       InstanceMethod("getDistance", &IndexWrapper::GetDistance),

       // New methods for Buffer/Stream support
       InstanceMethod("toBuffer", &IndexWrapper::ToBuffer),
       StaticMethod("fromBuffer", &IndexWrapper::FromBuffer),
       InstanceMethod("has", &IndexWrapper::Has),
       InstanceMethod("toString", &IndexWrapper::ToString),

       // Property accessors
       InstanceAccessor("space", &IndexWrapper::GetSpace, nullptr),
       InstanceAccessor("numDimensions", &IndexWrapper::GetNumDimensions,
                        nullptr),
       InstanceAccessor("M", &IndexWrapper::GetM, nullptr),
       InstanceAccessor("efConstruction", &IndexWrapper::GetEfConstruction,
                        nullptr),
       InstanceAccessor("maxElements", &IndexWrapper::GetMaxElements,
                        &IndexWrapper::SetMaxElements),
       InstanceAccessor("storageDataType", &IndexWrapper::GetStorageDataType,
                        nullptr),
       InstanceAccessor("numElements", &IndexWrapper::GetNumElements, nullptr),
       InstanceAccessor("ids", &IndexWrapper::GetIds, nullptr),
       InstanceAccessor("ef", &IndexWrapper::GetEf, &IndexWrapper::SetEf),
       InstanceAccessor("length", &IndexWrapper::GetLength, nullptr)});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();

  exports.Set("Index", func);
  return exports;
}

IndexWrapper::IndexWrapper(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<IndexWrapper>(info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsObject()) {
    Napi::TypeError::New(
        env, "Index() missing required argument: 'options' (an object)")
        .ThrowAsJavaScriptException();
    return;
  }

  Napi::Object options = info[0].As<Napi::Object>();

  // Extract required parameters
  if (!options.Has("space") || !options.Has("numDimensions")) {
    Napi::TypeError::New(
        env, "Index() missing required arguments: 'space' and 'numDimensions'")
        .ThrowAsJavaScriptException();
    return;
  }

  SpaceType space = static_cast<SpaceType>(
      options.Get("space").As<Napi::Number>().Uint32Value());
  int numDimensions =
      options.Get("numDimensions").As<Napi::Number>().Int32Value();

  // Extract optional parameters with defaults
  size_t M =
      options.Has("M") ? options.Get("M").As<Napi::Number>().Int64Value() : 12;
  size_t efConstruction =
      options.Has("efConstruction")
          ? options.Get("efConstruction").As<Napi::Number>().Int64Value()
          : 200;
  size_t randomSeed =
      options.Has("randomSeed")
          ? options.Get("randomSeed").As<Napi::Number>().Int64Value()
          : 1;
  size_t maxElements =
      options.Has("maxElements")
          ? options.Get("maxElements").As<Napi::Number>().Int64Value()
          : 1;
  StorageDataType storageDataType =
      options.Has("storageDataType")
          ? static_cast<StorageDataType>(
                options.Get("storageDataType").As<Napi::Number>().Uint32Value())
          : StorageDataType::Float32;

  // Create the appropriate typed index based on storage data type
  try {
    switch (storageDataType) {
    case StorageDataType::Float32:
      index_ = std::make_shared<TypedIndex<float>>(
          space, numDimensions, M, efConstruction, maxElements, randomSeed);
      break;
    case StorageDataType::Float8:
      index_ = std::make_shared<TypedIndex<float, int8_t, std::ratio<1, 127>>>(
          space, numDimensions, M, efConstruction, maxElements, randomSeed);
      break;
    case StorageDataType::E4M3:
      index_ = std::make_shared<TypedIndex<float, E4M3>>(
          space, numDimensions, M, efConstruction, maxElements, randomSeed);
      break;
    default:
      Napi::TypeError::New(env, "Unknown storage data type received.")
          .ThrowAsJavaScriptException();
      return;
    }
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return;
  }
}

// Helper method to convert Napi::Array to std::vector<float>
std::vector<float> IndexWrapper::ArrayToVector(const Napi::Array &arr) {
  std::vector<float> vec;
  vec.reserve(arr.Length());
  for (uint32_t i = 0; i < arr.Length(); i++) {
    vec.push_back(arr.Get(i).As<Napi::Number>().FloatValue());
  }
  return vec;
}

// Helper method to convert std::vector<float> to Napi::Array
Napi::Array IndexWrapper::VectorToArray(Napi::Env env,
                                        const std::vector<float> &vec) {
  Napi::Array arr = Napi::Array::New(env, vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    arr[i] = Napi::Number::New(env, vec[i]);
  }
  return arr;
}

Napi::Value IndexWrapper::AddItem(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(
        env, "addItem() missing required argument: 'vector' (an array)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Array vectorArray = info[0].As<Napi::Array>();
  std::vector<float> vector = ArrayToVector(vectorArray);

  std::optional<hnswlib::labeltype> id = std::nullopt;
  if (info.Length() >= 2 && info[1].IsNumber()) {
    id = info[1].As<Napi::Number>().Int64Value();
  }

  try {
    hnswlib::labeltype resultId = index_->addItem(vector, id);
    return Napi::Number::New(env, resultId);
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::AddItems(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(
        env,
        "addItems() missing required argument: 'vectors' (an array of arrays)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Array vectorsArray = info[0].As<Napi::Array>();
  std::vector<std::vector<float>> vectors;
  vectors.reserve(vectorsArray.Length());

  for (uint32_t i = 0; i < vectorsArray.Length(); i++) {
    if (!vectorsArray.Get(i).IsArray()) {
      Napi::TypeError::New(
          env, "addItems() expected each element of 'vectors' to be an array")
          .ThrowAsJavaScriptException();
      return env.Null();
    }
    Napi::Array vec = vectorsArray.Get(i).As<Napi::Array>();
    vectors.push_back(ArrayToVector(vec));
  }

  std::vector<hnswlib::labeltype> ids;
  if (info.Length() >= 2 && info[1].IsArray()) {
    Napi::Array idsArray = info[1].As<Napi::Array>();
    ids.reserve(idsArray.Length());
    for (uint32_t i = 0; i < idsArray.Length(); i++) {
      ids.push_back(idsArray.Get(i).As<Napi::Number>().Int64Value());
    }
  }

  int numThreads = -1;
  if (info.Length() >= 3 && info[2].IsNumber()) {
    numThreads = info[2].As<Napi::Number>().Int32Value();
  }

  try {
    std::vector<hnswlib::labeltype> resultIds =
        index_->addItems(vectors, ids, numThreads);
    Napi::Array result = Napi::Array::New(env, resultIds.size());
    for (size_t i = 0; i < resultIds.size(); i++) {
      result[i] = Napi::Number::New(env, resultIds[i]);
    }
    return result;
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::Query(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "query() missing required argument: 'vector' (an "
                              "array or an array of arrays)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int k = 1;
  if (info.Length() >= 2 && info[1].IsNumber()) {
    k = info[1].As<Napi::Number>().Int32Value();
  }

  int numThreads = -1;
  if (info.Length() >= 3 && info[2].IsNumber()) {
    numThreads = info[2].As<Napi::Number>().Int32Value();
  }

  long queryEf = -1;
  if (info.Length() >= 4 && info[3].IsNumber()) {
    queryEf = info[3].As<Napi::Number>().Int64Value();
  }

  Napi::Array inputArray = info[0].As<Napi::Array>();

  try {
    // Check if input is a single vector or multiple vectors
    bool isSingleVector =
        inputArray.Length() > 0 && inputArray.Get(uint32_t(0)).IsNumber();

    if (isSingleVector) {
      // Single query vector
      std::vector<float> queryVector = ArrayToVector(inputArray);
      auto [neighborIds, distances] = index_->query(queryVector, k, queryEf);

      Napi::Array result = Napi::Array::New(env);
      result.Set("neighbors",
                 VectorToArray(env, std::vector<float>(neighborIds.begin(),
                                                       neighborIds.end())));
      result.Set("distances", VectorToArray(env, distances));
      return result;
    } else {
      // Multiple query vectors
      std::vector<std::vector<float>> queryVectors;
      queryVectors.reserve(inputArray.Length());
      for (uint32_t i = 0; i < inputArray.Length(); i++) {
        if (!inputArray.Get(i).IsArray()) {
          Napi::TypeError::New(
              env, "query() expected one- or two-dimensional input data "
                   "(either a single query vector or multiple query vectors)")
              .ThrowAsJavaScriptException();
          return env.Null();
        }
        queryVectors.push_back(
            ArrayToVector(inputArray.Get(i).As<Napi::Array>()));
      }

      auto [neighborIds, distances] =
          index_->query(queryVectors, k, numThreads, queryEf);

      // Convevrt 2D NDArrays to JavaScript arrays
      Napi::Array neighborsResult = Napi::Array::New(env, neighborIds.shape[0]);
      Napi::Array distancesResult = Napi::Array::New(env, distances.shape[0]);

      for (size_t i = 0; i < neighborIds.shape[0]; i++) {
        Napi::Array neighborRow = Napi::Array::New(env, neighborIds.shape[1]);
        Napi::Array distanceRow = Napi::Array::New(env, distances.shape[1]);

        for (size_t j = 0; j < neighborIds.shape[1]; j++) {
          neighborRow[j] = Napi::Number::New(
              env, neighborIds.data[i * neighborIds.shape[1] + j]);
          distanceRow[j] = Napi::Number::New(
              env, distances.data[i * distances.shape[1] + j]);
        }

        neighborsResult[i] = neighborRow;
        distancesResult[i] = distanceRow;
      }

      Napi::Object result = Napi::Object::New(env);
      result.Set("neighbors", neighborsResult);
      result.Set("distances", distancesResult);
      return result;
    }
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::GetVector(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(
        env, "getVector() missing required argument: 'id' (a number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  hnswlib::labeltype id = info[0].As<Napi::Number>().Int64Value();

  try {
    std::vector<float> vector = index_->getVector(id);
    return VectorToArray(env, vector);
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::GetVectors(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(
        env,
        "getVectors() missing required argument: 'ids' (an array of numbers)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Array idsArray = info[0].As<Napi::Array>();
  std::vector<hnswlib::labeltype> ids;
  ids.reserve(idsArray.Length());

  for (uint32_t i = 0; i < idsArray.Length(); i++) {
    ids.push_back(idsArray.Get(i).As<Napi::Number>().Int64Value());
  }

  try {
    NDArray<float, 2> vectors = index_->getVectors(ids);

    Napi::Array result = Napi::Array::New(env, vectors.shape[0]);
    for (size_t i = 0; i < vectors.shape[0]; i++) {
      Napi::Array row = Napi::Array::New(env, vectors.shape[1]);
      for (size_t j = 0; j < vectors.shape[1]; j++) {
        row[j] = Napi::Number::New(env, vectors.data[i * vectors.shape[1] + j]);
      }
      result[i] = row;
    }

    return result;
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::MarkDeleted(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(
        env, "markDeleted() missing required argument: 'id' (a number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  hnswlib::labeltype id = info[0].As<Napi::Number>().Int64Value();

  try {
    index_->markDeleted(id);
    return env.Undefined();
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::UnmarkDeleted(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(
        env, "unmarkDeleted() missing required argument: 'id' (a number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  hnswlib::labeltype id = info[0].As<Napi::Number>().Int64Value();

  try {
    index_->unmarkDeleted(id);
    return env.Undefined();
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::Resize(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(
        env, "resize() missing required argument: 'newSize' (a number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  size_t newSize = info[0].As<Napi::Number>().Int64Value();

  try {
    index_->resizeIndex(newSize);
    return env.Undefined();
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::SaveIndex(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(
        env, "saveIndex() missing required argument: 'path' (a string)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string path = info[0].As<Napi::String>().Utf8Value();

  try {
    index_->saveIndex(path);
    return env.Undefined();
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::LoadIndex(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(
        env, "loadIndex() missing required argument: 'path' (a string)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::string path = info[0].As<Napi::String>().Utf8Value();

  // Optional parameters for loading legacy indices
  SpaceType space = SpaceType::Euclidean;
  int numDimensions = 0;
  StorageDataType storageDataType = StorageDataType::Float32;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object options = info[1].As<Napi::Object>();
    if (options.Has("space")) {
      space = static_cast<SpaceType>(
          options.Get("space").As<Napi::Number>().Uint32Value());
    }
    if (options.Has("numDimensions")) {
      numDimensions =
          options.Get("numDimensions").As<Napi::Number>().Int32Value();
    }
    if (options.Has("storageDataType")) {
      storageDataType = static_cast<StorageDataType>(
          options.Get("storageDataType").As<Napi::Number>().Uint32Value());
    }
  }

  try {
    auto inputStream = std::make_shared<FileInputStream>(path);

    // Try to load with metadata first
    std::unique_ptr<voyager::Metadata::V1> metadata =
        voyager::Metadata::loadFromStream(inputStream);

    std::shared_ptr<Index> loadedIndex;
    if (metadata) {
      // Modern index with metadata - validate if options were provided
      if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object options = info[1].As<Napi::Object>();
        if (options.Has("storageDataType")) {
          StorageDataType providedType = static_cast<StorageDataType>(
              options.Get("storageDataType").As<Napi::Number>().Uint32Value());
          if (metadata->getStorageDataType() != providedType) {
            std::string errorMsg =
                "Provided storage data type (" + toString(providedType) +
                ") does not match the data type used in this file (" +
                toString(metadata->getStorageDataType()) + ").";
            Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
            return env.Null();
          }
        }
        if (options.Has("space")) {
          SpaceType providedSpace = static_cast<SpaceType>(
              options.Get("space").As<Napi::Number>().Uint32Value());
          if (metadata->getSpaceType() != providedSpace) {
            std::string errorMsg =
                "Provided space type (" + toString(providedSpace) +
                ") does not match the space type used in this file (" +
                toString(metadata->getSpaceType()) + ").";
            Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
            return env.Null();
          }
        }
        if (options.Has("numDimensions")) {
          int providedDims =
              options.Get("numDimensions").As<Napi::Number>().Uint32Value();
          if (metadata->getNumDimensions() != providedDims) {
            std::string errorMsg =
                "Provided number of dimensions (" +
                std::to_string(providedDims) +
                ") does not match the number of dimensions used in this file "
                "(" +
                std::to_string(metadata->getNumDimensions()) + ").";
            Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
            return env.Null();
          }
        }
      }
      loadedIndex =
          loadTypedIndexFromMetadata(std::move(metadata), inputStream);
    } else {
      // Legacy index without metadata - need explicit parameters
      if (numDimensions <= 0) {
        Napi::TypeError::New(
            env, "Index file has no metadata. Please provide space, "
                 "numDimensions, and storageDataType options.")
            .ThrowAsJavaScriptException();
        return env.Null();
      }

      switch (storageDataType) {
      case StorageDataType::Float32:
        loadedIndex = std::make_shared<TypedIndex<float>>(inputStream, space,
                                                          numDimensions);
        break;
      case StorageDataType::Float8:
        loadedIndex =
            std::make_shared<TypedIndex<float, int8_t, std::ratio<1, 127>>>(
                inputStream, space, numDimensions);
        break;
      case StorageDataType::E4M3:
        loadedIndex = std::make_shared<TypedIndex<float, E4M3>>(
            inputStream, space, numDimensions);
        break;
      default:
        Napi::TypeError::New(env, "Unknown storage data type received.")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
    }

    // Create a new IndexWrapper instance with the loaded index
    // We need to creare a dummy options object to pass to the constructor
    // since we're bypassing normal initialization
    Napi::Object dummyOptions = Napi::Object::New(env);
    dummyOptions.Set(
        "space",
        Napi::Number::New(env, static_cast<int>(loadedIndex->getSpace())));
    dummyOptions.Set("numDimensions",
                     Napi::Number::New(env, loadedIndex->getNumDimensions()));

    Napi::Object instance = constructor.New({dummyOptions});
    IndexWrapper *wrapper = Napi::ObjectWrap<IndexWrapper>::Unwrap(instance);
    wrapper->index_ = loadedIndex;

    return instance;
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::GetDistance(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsArray()) {
    Napi::TypeError::New(
        env,
        "getDistance() missing required arguments: 'a' and 'b' (both arrays)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  std::vector<float> a = ArrayToVector(info[0].As<Napi::Array>());
  std::vector<float> b = ArrayToVector(info[1].As<Napi::Array>());

  try {
    float distance = index_->getDistance(a, b);
    return Napi::Number::New(env, distance);
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// Property getters/setters
Napi::Value IndexWrapper::GetSpace(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, static_cast<int>(index_->getSpace()));
}

Napi::Value IndexWrapper::GetNumDimensions(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, index_->getNumDimensions());
}

Napi::Value IndexWrapper::GetM(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, index_->getM());
}

Napi::Value IndexWrapper::GetEfConstruction(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, index_->getEfConstruction());
}

Napi::Value IndexWrapper::GetMaxElements(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, index_->getMaxElements());
}

void IndexWrapper::SetMaxElements(const Napi::CallbackInfo &info,
                                  const Napi::Value &value) {
  Napi::Env env = info.Env();

  if (!value.IsNumber()) {
    Napi::TypeError::New(env, "maxElements must be set to a number")
        .ThrowAsJavaScriptException();
    return;
  }

  size_t newSize = value.As<Napi::Number>().Int64Value();

  try {
    index_->resizeIndex(newSize);
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
}

Napi::Value IndexWrapper::GetStorageDataType(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, static_cast<int>(index_->getStorageDataType()));
}

Napi::Value IndexWrapper::GetNumElements(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, index_->getNumElements());
}

Napi::Value IndexWrapper::GetIds(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  try {
    std::vector<hnswlib::labeltype> ids = index_->getIDs();
    Napi::Array result = Napi::Array::New(env, ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
      result[i] = Napi::Number::New(env, ids[i]);
    }
    return result;
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::GetEf(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, index_->getEF());
}

void IndexWrapper::SetEf(const Napi::CallbackInfo &info,
                         const Napi::Value &value) {
  Napi::Env env = info.Env();

  if (!value.IsNumber()) {
    Napi::TypeError::New(env, "ef must be set to a number")
        .ThrowAsJavaScriptException();
    return;
  }

  size_t ef = value.As<Napi::Number>().Int64Value();

  try {
    index_->setEF(ef);
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
}

Napi::Value IndexWrapper::GetLength(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, index_->getIDsMap().size());
}

// New methods for Buffer/Stream support
Napi::Value IndexWrapper::ToBuffer(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  try {
    auto outputStream = std::make_shared<MemoryOutputStream>();
    index_->saveIndex(outputStream);
    std::string bytes = outputStream->getValue();
    Napi::Buffer<char> buffer =
        Napi::Buffer<char>::Copy(env, bytes.data(), bytes.size());
    return buffer;
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::FromBuffer(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsBuffer()) {
    Napi::TypeError::New(
        env, "fromBuffer() missing required argument: 'buffer' (a Buffer)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  Napi::Buffer<char> buffer = info[0].As<Napi::Buffer<char>>();
  std::string data(buffer.Data(), buffer.Length());

  // Optional parameters for loading legacy indices
  SpaceType space = SpaceType::Euclidean;
  int numDimensions = 0;
  StorageDataType storageDataType = StorageDataType::Float32;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object options = info[1].As<Napi::Object>();
    if (options.Has("space")) {
      space = static_cast<SpaceType>(
          options.Get("space").As<Napi::Number>().Uint32Value());
    }
    if (options.Has("numDimensions")) {
      numDimensions =
          options.Get("numDimensions").As<Napi::Number>().Int32Value();
    }
    if (options.Has("storageDataType")) {
      storageDataType = static_cast<StorageDataType>(
          options.Get("storageDataType").As<Napi::Number>().Uint32Value());
    }
  }

  try {
    auto inputStream = std::make_shared<MemoryInputStream>(data);

    // Try to load with metadata first
    std::unique_ptr<voyager::Metadata::V1> metadata =
        voyager::Metadata::loadFromStream(inputStream);

    std::shared_ptr<Index> loadedIndex;
    if (metadata) {
      // Modern index with metadata - validate if options were provided
      if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object options = info[1].As<Napi::Object>();
        if (options.Has("storageDataType")) {
          StorageDataType providedType = static_cast<StorageDataType>(
              options.Get("storageDataType").As<Napi::Number>().Uint32Value());
          if (metadata->getStorageDataType() != providedType) {
            std::string errorMsg =
                "Provided storage data type (" + toString(providedType) +
                ") does not match the data type used in this file (" +
                toString(metadata->getStorageDataType()) + ").";
            Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
            return env.Null();
          }
        }
        if (options.Has("space")) {
          SpaceType providedSpace = static_cast<SpaceType>(
              options.Get("space").As<Napi::Number>().Uint32Value());
          if (metadata->getSpaceType() != providedSpace) {
            std::string errorMsg =
                "Provided space type (" + toString(providedSpace) +
                ") does not match the space type used in this file (" +
                toString(metadata->getSpaceType()) + ").";
            Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
            return env.Null();
          }
        }
        if (options.Has("numDimensions")) {
          int providedDims =
              options.Get("numDimensions").As<Napi::Number>().Uint32Value();
          if (metadata->getNumDimensions() != providedDims) {
            std::string errorMsg =
                "Provided number of dimensions (" +
                std::to_string(providedDims) +
                ") does not match the number of dimensions used in this file "
                "(" +
                std::to_string(metadata->getNumDimensions()) + ").";
            Napi::Error::New(env, errorMsg).ThrowAsJavaScriptException();
            return env.Null();
          }
        }
      }
      loadedIndex =
          loadTypedIndexFromMetadata(std::move(metadata), inputStream);
    } else {
      // Legacy index without metadata - need explicit parameters
      if (numDimensions == 0) {
        Napi::TypeError::New(
            env, "Index buffer has no metadata. Please provide space, "
                 "numDimensions, and storageDataType options.")
            .ThrowAsJavaScriptException();
        return env.Null();
      }

      switch (storageDataType) {
      case StorageDataType::Float32:
        loadedIndex = std::make_shared<TypedIndex<float>>(inputStream, space,
                                                          numDimensions);
        break;
      case StorageDataType::Float8:
        loadedIndex =
            std::make_shared<TypedIndex<float, int8_t, std::ratio<1, 127>>>(
                inputStream, space, numDimensions);
        break;
      case StorageDataType::E4M3:
        loadedIndex = std::make_shared<TypedIndex<float, E4M3>>(
            inputStream, space, numDimensions);
        break;
      default:
        Napi::TypeError::New(env, "Unknown storage data type received.")
            .ThrowAsJavaScriptException();
        return env.Null();
      }
    }

    // Create a new IndexWrapper instance with the loaded index
    // We need to creare a dummy options object to pass to the constructor
    // since we're bypassing normal initialization
    Napi::Object dummyOptions = Napi::Object::New(env);
    dummyOptions.Set(
        "space",
        Napi::Number::New(env, static_cast<int>(loadedIndex->getSpace())));
    dummyOptions.Set("numDimensions",
                     Napi::Number::New(env, loadedIndex->getNumDimensions()));

    Napi::Object instance = constructor.New({dummyOptions});
    IndexWrapper *wrapper = Napi::ObjectWrap<IndexWrapper>::Unwrap(instance);
    wrapper->index_ = loadedIndex;

    return instance;
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::Has(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env,
                         "has() missing required argument: 'id' (a number)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  hnswlib::labeltype id = info[0].As<Napi::Number>().Int64Value();

  try {
    auto &map = index_->getIDsMap();
    return Napi::Boolean::New(env, map.find(id) != map.end());
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

Napi::Value IndexWrapper::ToString(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  try {
    std::ostringstream oss;
    oss << "Index(space=" << index_->getSpaceName()
        << ", dimensions=" << index_->getNumDimensions()
        << ", storageDatatType=" << index_->getStorageDataType()
        << ", M=" << index_->getM()
        << ", efConstruction=" << index_->getEfConstruction()
        << ", numElements=" << index_->getNumElements()
        << ", maxElements=" << index_->getMaxElements() << ")";
    return Napi::String::New(env, oss.str());
  } catch (const std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return env.Null();
  }
}

// Space enum initialization
Napi::Object InitSpace(Napi::Env env) {
  Napi::Object space = Napi::Object::New(env);
  space.Set("Euclidean", Napi::Number::New(
                             env, static_cast<uint32_t>(SpaceType::Euclidean)));
  space.Set("Cosine",
            Napi::Number::New(env, static_cast<uint32_t>(SpaceType::Cosine)));
  space.Set(
      "InnerProduct",
      Napi::Number::New(env, static_cast<uint32_t>(SpaceType::InnerProduct)));
  return space;
}

// StorageDataType enum initialization
Napi::Object InitStorageDataType(Napi::Env env) {
  Napi::Object storageDataType = Napi::Object::New(env);
  storageDataType.Set(
      "Float8",
      Napi::Number::New(env, static_cast<uint32_t>(StorageDataType::Float8)));
  storageDataType.Set(
      "Float32",
      Napi::Number::New(env, static_cast<uint32_t>(StorageDataType::Float32)));
  storageDataType.Set(
      "E4M3",
      Napi::Number::New(env, static_cast<uint32_t>(StorageDataType::E4M3)));
  return storageDataType;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // Export the Index class
  IndexWrapper::Init(env, exports);

  // Export the Space enum
  exports.Set("Space", InitSpace(env));

  // Export the StorageDataType enum
  exports.Set("StorageDataType", InitStorageDataType(env));

  return exports;
}

NODE_API_MODULE(voyager_node, Init)
