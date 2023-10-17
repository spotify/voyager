/*-
 * -\-\-
 * voyager
 * --
 * Copyright (C) 2016 - 2023 Spotify AB
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

#include "com_spotify_voyager_jni_Index.h"
#include "JavaInputStream.h"
#include "JavaOutputStream.h"
#include <Enums.h>
#include <Index.h>
#include <TypedIndex.h>

#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

/**
 * Given a Java object, return the field ID for the "native handle" property
 * within that object, which can be used to store a C++ pointer.
 */
jfieldID getHandleFieldID(JNIEnv *env, jobject obj) {
  jclass c = env->GetObjectClass(obj);
  // J is the type signature for long:
  return env->GetFieldID(c, "nativeHandle", "J");
}

template <typename T>
T *getHandle(JNIEnv *env, jobject obj, bool allow_missing = false) {
  jlong handle = env->GetLongField(obj, getHandleFieldID(env, obj));
  T *pointer = reinterpret_cast<T *>(handle);

  if (!allow_missing && !pointer) {
    throw std::runtime_error("Native JNI object not found.");
  }

  return pointer;
}

template <typename T> void setHandle(JNIEnv *env, jobject obj, T *t) {
  env->SetLongField(obj, getHandleFieldID(env, obj),
                    reinterpret_cast<jlong>(t));
}

std::string toString(JNIEnv *env, jstring js) {
  std::string result;
  long len = env->GetStringUTFLength(js);
  result.resize(len);
  if (len > 0) {
    env->GetStringUTFRegion(js, 0, len, result.data());
  }
  return result;
}

std::string toString(JNIEnv *env, jobject object) {
  jclass javaClass = env->GetObjectClass(object);
  if (javaClass == 0) {
    throw std::runtime_error(
        "C++ bindings were unable to get the class for the provided object.");
  }

  return toString(env, (jstring)env->CallObjectMethod(
                           object, env->GetMethodID(javaClass, "toString",
                                                    "()Ljava/lang/String;")));
}

SpaceType toSpaceType(JNIEnv *env, jobject enumVal) {
  std::string enumValueName = toString(env, enumVal);

  // TODO: Replace me with a usage of MagicEnum!
  if (enumValueName == "Euclidean") {
    return SpaceType::Euclidean;
  } else if (enumValueName == "InnerProduct") {
    return SpaceType::InnerProduct;
  } else if (enumValueName == "Cosine") {
    return SpaceType::Cosine;
  } else {
    throw std::runtime_error(
        "Voyager C++ bindings received unknown enum value \"" + enumValueName +
        "\".");
  }
}

jobject toSpaceType(JNIEnv *env, SpaceType enumVal) {
  jclass enumClass = env->FindClass("com/spotify/voyager/jni/Index$SpaceType");
  if (!enumClass) {
    throw std::runtime_error(
        "C++ bindings could not find SpaceType Java enum!");
  }

  const char *enumValueName = nullptr;

  switch (enumVal) {
  case SpaceType::Euclidean:
    enumValueName = "Euclidean";
    break;
  case SpaceType::InnerProduct:
    enumValueName = "InnerProduct";
    break;
  case SpaceType::Cosine:
    enumValueName = "Cosine";
    break;
  default:
    throw std::runtime_error(
        "Voyager C++ bindings received unknown enum value.");
  }

  jfieldID fieldID = env->GetStaticFieldID(
      enumClass, enumValueName, "Lcom/spotify/voyager/jni/Index$SpaceType;");
  if (!fieldID) {
    throw std::runtime_error(
        "C++ bindings could not find value in SpaceType Java enum!");
  }

  jobject javaValue = env->GetStaticObjectField(enumClass, fieldID);
  if (!javaValue) {
    throw std::runtime_error("C++ bindings could not find static object field "
                             "for in SpaceType Java enum!");
  }

  return javaValue;
}

StorageDataType toStorageDataType(JNIEnv *env, jobject enumVal) {
  std::string enumValueName = toString(env, enumVal);

  // TODO: Replace me with a usage of MagicEnum!
  if (enumValueName == "Float8") {
    return StorageDataType::Float8;
  } else if (enumValueName == "Float32") {
    return StorageDataType::Float32;
  } else if (enumValueName == "E4M3") {
    return StorageDataType::E4M3;
  } else {
    throw std::runtime_error(
        "Voyager C++ bindings received unknown enum value \"" + enumValueName +
        "\".");
  }
}

jobject toStorageDataType(JNIEnv *env, StorageDataType enumVal) {
  jclass enumClass =
      env->FindClass("com/spotify/voyager/jni/Index$StorageDataType");

  if (!enumClass) {
    throw std::runtime_error(
        "C++ bindings could not find StorageDataType Java enum!");
  }

  const char *enumValueName = nullptr;

  switch (enumVal) {
  case StorageDataType::Float8:
    enumValueName = "Float8";
    break;
  case StorageDataType::Float32:
    enumValueName = "Float32";
    break;
  case StorageDataType::E4M3:
    enumValueName = "E4M3";
    break;
  default:
    throw std::runtime_error(
        "Voyager C++ bindings received unknown enum value.");
  }

  jfieldID fieldID =
      env->GetStaticFieldID(enumClass, enumValueName,
                            "Lcom/spotify/voyager/jni/Index$StorageDataType;");
  if (!fieldID) {
    throw std::runtime_error(
        "C++ bindings could not find value in StorageDataType Java enum!");
  }

  jobject javaValue = env->GetStaticObjectField(enumClass, fieldID);
  if (!javaValue) {
    throw std::runtime_error("C++ bindings could not find static object field "
                             "for in StorageDataType Java enum!");
  }

  return javaValue;
}

/**
 * Convert a Java nested array (array of float arrays) to a 2D NDArray.
 */
NDArray<float, 2> toNDArray(JNIEnv *env, jobjectArray floatArrays) {
  jsize numElements = env->GetArrayLength(floatArrays);
  if (numElements == 0) {
    return NDArray<float, 2>({0, 0});
  }

  jobject firstElement = env->GetObjectArrayElement(floatArrays, 0);
  jsize numDimensions = env->GetArrayLength((jfloatArray)firstElement);
  env->DeleteLocalRef(firstElement);
  if (numDimensions == 0) {
    return NDArray<float, 2>({0, 0});
  }

  NDArray<float, 2> output = NDArray<float, 2>({numElements, numDimensions});

  float *outputPointer = output.data.data();
  for (int i = 0; i < numElements; i++) {
    jobject element = env->GetObjectArrayElement(floatArrays, i);

    // TODO: Ensure that each element is actually a float array!
    jfloatArray floatArray = (jfloatArray)element;

    jsize numVectorDimensions = env->GetArrayLength(floatArray);

    if (numVectorDimensions != numDimensions) {
      throw std::runtime_error("When passing an array of arrays, all "
                               "sub-arrays must be the same length.");
    }

    env->GetFloatArrayRegion(floatArray, 0, numDimensions, outputPointer);

    // Delete the local reference to the nested float array.
    // Note that this isn't necessary; it merely helps the Java GC
    // identify if/when these elements can be cleaned up earlier than when
    // this function returns.
    // Removing this call would not create a memory leak.
    env->DeleteLocalRef(element);

    outputPointer += numDimensions;
  }

  return output;
}

/**
 * Convert a Java float array to a std::vector<float>.
 */
std::vector<float> toStdVector(JNIEnv *env, jfloatArray floatArray) {
  jsize numElements = env->GetArrayLength(floatArray);
  std::vector<float> input(numElements);
  env->GetFloatArrayRegion(floatArray, 0, numElements, (float *)input.data());
  return input;
}

/**
 * Convert a std::vector<float> to a Java float array.
 */
jfloatArray toFloatArray(JNIEnv *env, std::vector<float> floatArray) {
  jfloatArray returnArray = env->NewFloatArray(floatArray.size());
  env->SetFloatArrayRegion(returnArray, 0, floatArray.size(),
                           floatArray.data());
  return returnArray;
}

/**
 * Convert a Java long array to a std::vector<size_t>.
 * Note that this function will underflow if any elements are negative.
 */
std::vector<size_t> toUnsignedStdVector(JNIEnv *env, jlongArray longArray) {
  jsize numElements = env->GetArrayLength(longArray);
  std::vector<size_t> input(numElements);
  env->GetLongArrayRegion(longArray, 0, numElements, (jlong *)input.data());
  return input;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Index Construction and Indexing
////////////////////////////////////////////////////////////////////////////////////////////////////
void Java_com_spotify_voyager_jni_Index_nativeConstructor(
    JNIEnv *env, jobject self, jobject spaceType, jint numDimensions, jlong M,
    jlong efConstruction, jlong randomSeed, jlong maxElements,
    jobject storageDataType) {

  try {
    switch (toStorageDataType(env, storageDataType)) {
    case StorageDataType::Float32:
      setHandle<Index>(env, self,
                       new TypedIndex<float>(toSpaceType(env, spaceType),
                                             numDimensions, M, efConstruction,
                                             randomSeed, maxElements));
      break;
    case StorageDataType::Float8:
      setHandle<Index>(env, self,
                       new TypedIndex<float, int8_t, std::ratio<1, 127>>(
                           toSpaceType(env, spaceType), numDimensions, M,
                           efConstruction, randomSeed, maxElements));
      break;
    case StorageDataType::E4M3:
      setHandle<Index>(env, self,
                       new TypedIndex<float, E4M3>(
                           toSpaceType(env, spaceType), numDimensions, M,
                           efConstruction, randomSeed, maxElements));
      break;
    }
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

void Java_com_spotify_voyager_jni_Index_addItem___3F(JNIEnv *env, jobject self,
                                                     jfloatArray vector) {
  try {
    Index *index = getHandle<Index>(env, self);
    index->addItem(toStdVector(env, vector), {});
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

void Java_com_spotify_voyager_jni_Index_addItem___3FJ(JNIEnv *env, jobject self,
                                                      jfloatArray vector,
                                                      jlong id) {
  try {
    Index *index = getHandle<Index>(env, self);
    index->addItem(toStdVector(env, vector), {id});
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

void Java_com_spotify_voyager_jni_Index_addItems___3_3FI(JNIEnv *env,
                                                         jobject self,
                                                         jobjectArray vectors,
                                                         jint numThreads) {
  try {
    Index *index = getHandle<Index>(env, self);
    index->addItems(toNDArray(env, vectors), {}, numThreads);
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

void Java_com_spotify_voyager_jni_Index_addItems___3_3F_3JI(
    JNIEnv *env, jobject self, jobjectArray vectors, jlongArray ids,
    jint numThreads) {
  try {
    Index *index = getHandle<Index>(env, self);
    index->addItems(toNDArray(env, vectors), toUnsignedStdVector(env, ids),
                    numThreads);
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Querying
////////////////////////////////////////////////////////////////////////////////////////////////////
jobject Java_com_spotify_voyager_jni_Index_query___3FIJ(JNIEnv *env,
                                                        jobject self,
                                                        jfloatArray queryVector,
                                                        jint numNeighbors,
                                                        jlong queryEf) {
  try {
    Index *index = getHandle<Index>(env, self);

    std::tuple<std::vector<hnswlib::labeltype>, std::vector<float>>
        queryResults =
            index->query(toStdVector(env, queryVector), numNeighbors, queryEf);

    jclass queryResultsClass =
        env->FindClass("com/spotify/voyager/jni/Index$QueryResults");
    if (!queryResultsClass) {
      throw std::runtime_error(
          "C++ bindings failed to find QueryResults class.");
    }

    jmethodID constructor =
        env->GetMethodID(queryResultsClass, "<init>", "([J[F)V");

    if (!constructor) {
      throw std::runtime_error(
          "C++ bindings failed to find QueryResults constructor.");
    }

    // Allocate a Java long array for the IDs:
    jlongArray labels = env->NewLongArray(numNeighbors);

    // queryResults is a (size_t *), but labels is a signed (long *).
    //  This may overflow if we have more than... 2^63 = 9.223372037e18
    //  elements. We're probably safe doing this.
    env->SetLongArrayRegion(labels, 0, numNeighbors,
                            (jlong *)std::get<0>(queryResults).data());

    jfloatArray distances = env->NewFloatArray(numNeighbors);
    env->SetFloatArrayRegion(distances, 0, numNeighbors,
                             std::get<1>(queryResults).data());

    return env->NewObject(queryResultsClass, constructor, labels, distances);
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
    return nullptr;
  }
}

jobjectArray Java_com_spotify_voyager_jni_Index_query___3_3FIIJ(
    JNIEnv *env, jobject self, jobjectArray queryVectors, jint numNeighbors,
    jint numThreads, jlong queryEf) {
  try {
    Index *index = getHandle<Index>(env, self);

    int numQueries = env->GetArrayLength(queryVectors);

    std::tuple<NDArray<hnswlib::labeltype, 2>, NDArray<float, 2>> queryResults =
        index->query(toNDArray(env, queryVectors), numNeighbors, numThreads,
                     queryEf);

    jclass queryResultsClass =
        env->FindClass("com/spotify/voyager/jni/Index$QueryResults");
    if (!queryResultsClass) {
      throw std::runtime_error(
          "C++ bindings failed to find QueryResults class.");
    }

    jmethodID constructor =
        env->GetMethodID(queryResultsClass, "<init>", "([J[F)V");

    if (!constructor) {
      throw std::runtime_error(
          "C++ bindings failed to find QueryResults constructor.");
    }

    jobjectArray javaQueryResults =
        env->NewObjectArray(numQueries, queryResultsClass, NULL);

    for (int i = 0; i < numQueries; i++) {
      // Allocate a Java long array for the indices, and a float array for the
      // distances:
      jlongArray labels = env->NewLongArray(numNeighbors);

      // queryResults is a (size_t *), but labels is a signed (long *).
      //  This may overflow if we have more than... 2^63 = 9.223372037e18
      //  elements. We're probably safe doing this.
      env->SetLongArrayRegion(labels, 0, numNeighbors,
                              (jlong *)std::get<0>(queryResults)[i]);

      jfloatArray distances = env->NewFloatArray(numNeighbors);
      env->SetFloatArrayRegion(distances, 0, numNeighbors,
                               std::get<1>(queryResults)[i]);

      jobject queryResults =
          env->NewObject(queryResultsClass, constructor, labels, distances);
      env->SetObjectArrayElement(javaQueryResults, i, queryResults);
      env->DeleteLocalRef(labels);
      env->DeleteLocalRef(distances);
      env->DeleteLocalRef(queryResults);
    }

    return javaQueryResults;
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
    return nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Property Methods
////////////////////////////////////////////////////////////////////////////////////////////////////
jobject Java_com_spotify_voyager_jni_Index_getSpace(JNIEnv *env, jobject self) {
  try {
    return toSpaceType(env, getHandle<Index>(env, self)->getSpace());
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
  return nullptr;
}

jint Java_com_spotify_voyager_jni_Index_getNumDimensions(JNIEnv *env,
                                                         jobject self) {
  try {
    return getHandle<Index>(env, self)->getNumDimensions();
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
  return 0;
}

jlong Java_com_spotify_voyager_jni_Index_getM(JNIEnv *env, jobject self) {
  try {
    return getHandle<Index>(env, self)->getM();
  } catch (std::exception const &e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
  }
  return 0;
}

jlong Java_com_spotify_voyager_jni_Index_getEfConstruction(JNIEnv *env,
                                                           jobject self) {
  try {
    return getHandle<Index>(env, self)->getEfConstruction();
  } catch (std::exception const &e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
  }
  return 0;
}

jlong Java_com_spotify_voyager_jni_Index_getMaxElements(JNIEnv *env,
                                                        jobject self) {
  try {
    return getHandle<Index>(env, self)->getMaxElements();
  } catch (std::exception const &e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
  }
  return 0;
}

jobject Java_com_spotify_voyager_jni_Index_getStorageDataType(JNIEnv *env,
                                                              jobject self) {
  try {
    return toStorageDataType(env,
                             getHandle<Index>(env, self)->getStorageDataType());
  } catch (std::exception const &e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
  }
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Index Accessor Methods
////////////////////////////////////////////////////////////////////////////////////////////////////
jlong Java_com_spotify_voyager_jni_Index_getNumElements(JNIEnv *env,
                                                        jobject self) {
  try {
    return getHandle<Index>(env, self)->getNumElements();
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
  return 0;
}

jfloatArray Java_com_spotify_voyager_jni_Index_getVector(JNIEnv *env,
                                                         jobject self,
                                                         jlong id) {
  try {
    Index *index = getHandle<Index>(env, self);
    return toFloatArray(env, index->getVector(id));
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
    return nullptr;
  }
}

jobjectArray Java_com_spotify_voyager_jni_Index_getVectors(JNIEnv *env,
                                                           jobject self,
                                                           jlongArray ids) {
  try {
    Index *index = getHandle<Index>(env, self);

    NDArray<float, 2> vectors =
        index->getVectors(toUnsignedStdVector(env, ids));

    jclass floatArrayClass = env->FindClass("[F");
    if (!floatArrayClass) {
      throw std::runtime_error("C++ bindings failed to find float[] class.");
    }

    jobjectArray javaVectors =
        env->NewObjectArray(vectors.shape[0], floatArrayClass, NULL);

    for (int i = 0; i < vectors.shape[0]; i++) {
      jfloatArray vector = env->NewFloatArray(vectors.shape[1]);
      env->SetFloatArrayRegion(vector, 0, vectors.shape[1], vectors[i]);
      env->SetObjectArrayElement(javaVectors, i, vector);
      env->DeleteLocalRef(vector);
    }

    return javaVectors;
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
    return nullptr;
  }
}

jlongArray Java_com_spotify_voyager_jni_Index_getIDs(JNIEnv *env,
                                                     jobject self) {
  try {
    Index *index = getHandle<Index>(env, self);

    std::vector<hnswlib::labeltype> ids = index->getIDs();

    static_assert(sizeof(hnswlib::labeltype) == sizeof(jlong),
                  "getIDs expects hnswlib::labeltype to be a 64-bit integer.");

    jclass longArrayClass = env->FindClass("[J");
    if (!longArrayClass) {
      throw std::runtime_error("C++ bindings failed to find long[] class.");
    }

    // Allocate a Java long array for the IDs:
    jlongArray javaIds = env->NewLongArray(ids.size());

    env->SetLongArrayRegion(javaIds, 0, ids.size(), (jlong *)ids.data());

    return javaIds;
  } catch (std::exception const &e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Index Modifier Methods
////////////////////////////////////////////////////////////////////////////////////////////////////
void Java_com_spotify_voyager_jni_Index_setEf(JNIEnv *env, jobject self,
                                              jlong newEf) {
  try {
    getHandle<Index>(env, self)->setEF(newEf);
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

jint Java_com_spotify_voyager_jni_Index_getEf(JNIEnv *env, jobject self) {
  try {
    return getHandle<Index>(env, self)->getEF();
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
  return 0;
}

// TODO: Add markDeleted

// TODO: Add unmarkDeleted

// TODO: Add resizeIndex

////////////////////////////////////////////////////////////////////////////////////////////////////
// Save Index
////////////////////////////////////////////////////////////////////////////////////////////////////
void Java_com_spotify_voyager_jni_Index_saveIndex__Ljava_lang_String_2(
    JNIEnv *env, jobject self, jstring filename) {
  try {
    Index *index = getHandle<Index>(env, self);
    index->saveIndex(toString(env, filename));
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

void Java_com_spotify_voyager_jni_Index_saveIndex__Ljava_io_OutputStream_2(
    JNIEnv *env, jobject self, jobject outputStream) {
  try {
    Index *index = getHandle<Index>(env, self);
    index->saveIndex(std::make_shared<JavaOutputStream>(env, outputStream));
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

// TODO: Add asBytes

////////////////////////////////////////////////////////////////////////////////////////////////////
// Load Index
////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: Convert these to static methods
void Java_com_spotify_voyager_jni_Index_nativeLoadFromFileWithParameters(
    JNIEnv *env, jobject self, jstring filename, jobject spaceType,
    jint numDimensions, jobject storageDataType) {
  try {
    auto inputStream =
        std::make_shared<FileInputStream>(toString(env, filename));
    std::unique_ptr<voyager::Metadata::V1> metadata =
        voyager::Metadata::loadFromStream(inputStream);

    if (metadata) {
      if (metadata->getStorageDataType() !=
          toStorageDataType(env, storageDataType)) {
        throw std::domain_error(
            "Provided storage data type (" +
            toString(toStorageDataType(env, storageDataType)) +
            ") does not match the data type used in this file (" +
            toString(metadata->getStorageDataType()) + ").");
      }
      if (metadata->getSpaceType() != toSpaceType(env, spaceType)) {
        throw std::domain_error(
            "Provided space type (" + toString(toSpaceType(env, spaceType)) +
            ") does not match the space type used in this file (" +
            toString(metadata->getSpaceType()) + ").");
      }
      if (metadata->getNumDimensions() != numDimensions) {
        throw std::domain_error(
            "Provided number of dimensions (" + std::to_string(numDimensions) +
            ") does not match the number of dimensions used in this file (" +
            std::to_string(metadata->getNumDimensions()) + ").");
      }
    }

    switch (toStorageDataType(env, storageDataType)) {
    case StorageDataType::Float32:
      setHandle<Index>(env, self,
                       new TypedIndex<float>(inputStream,
                                             toSpaceType(env, spaceType),
                                             numDimensions));
      break;
    case StorageDataType::Float8:
      setHandle<Index>(
          env, self,
          new TypedIndex<float, int8_t, std::ratio<1, 127>>(
              inputStream, toSpaceType(env, spaceType), numDimensions));
      break;
    case StorageDataType::E4M3:
      setHandle<Index>(env, self,
                       new TypedIndex<float, E4M3>(inputStream,
                                                   toSpaceType(env, spaceType),
                                                   numDimensions));
      break;
    }
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

void Java_com_spotify_voyager_jni_Index_nativeLoadFromInputStreamWithParameters(
    JNIEnv *env, jobject self, jobject jInputStream, jobject spaceType,
    jint numDimensions, jobject storageDataType) {
  try {
    auto inputStream = std::make_shared<JavaInputStream>(env, jInputStream);
    std::unique_ptr<voyager::Metadata::V1> metadata =
        voyager::Metadata::loadFromStream(inputStream);

    if (metadata) {
      if (metadata->getStorageDataType() !=
          toStorageDataType(env, storageDataType)) {
        throw std::domain_error(
            "Provided storage data type (" +
            toString(toStorageDataType(env, storageDataType)) +
            ") does not match the data type used in this file (" +
            toString(metadata->getStorageDataType()) + ").");
      }
      if (metadata->getSpaceType() != toSpaceType(env, spaceType)) {
        throw std::domain_error(
            "Provided space type (" + toString(toSpaceType(env, spaceType)) +
            ") does not match the space type used in this file (" +
            toString(metadata->getSpaceType()) + ").");
      }
      if (metadata->getNumDimensions() != numDimensions) {
        throw std::domain_error(
            "Provided number of dimensions (" + std::to_string(numDimensions) +
            ") does not match the number of dimensions used in this file (" +
            std::to_string(metadata->getNumDimensions()) + ").");
      }
    }

    switch (toStorageDataType(env, storageDataType)) {
    case StorageDataType::Float32:
      if (metadata) {
        setHandle<Index>(env, self,
                         new TypedIndex<float>(metadata, inputStream,
                                               toSpaceType(env, spaceType),
                                               numDimensions));
      } else {
        setHandle<Index>(env, self,
                         new TypedIndex<float>(inputStream,
                                               toSpaceType(env, spaceType),
                                               numDimensions));
      }
      break;
    case StorageDataType::Float8:
      if (metadata) {
        setHandle<Index>(env, self,
                         new TypedIndex<float, int8_t, std::ratio<1, 127>>(
                             metadata, inputStream, toSpaceType(env, spaceType),
                             numDimensions));

      } else {
        setHandle<Index>(
            env, self,
            new TypedIndex<float, int8_t, std::ratio<1, 127>>(
                inputStream, toSpaceType(env, spaceType), numDimensions));
      }
      break;
    case StorageDataType::E4M3:
      if (metadata) {
        setHandle<Index>(env, self,
                         new TypedIndex<float, E4M3>(
                             metadata, inputStream, toSpaceType(env, spaceType),
                             numDimensions));
      } else {
        setHandle<Index>(
            env, self,
            new TypedIndex<float, E4M3>(
                inputStream, toSpaceType(env, spaceType), numDimensions));
      }
      break;
    }
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}

void Java_com_spotify_voyager_jni_Index_nativeDestructor(JNIEnv *env,
                                                         jobject self) {
  try {
    if (Index *index = getHandle<Index>(env, self, true)) {
      delete index;
      setHandle<Index>(env, self, nullptr);
    }
  } catch (std::exception const &e) {
    if (!env->ExceptionCheck()) {
      env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
  }
}