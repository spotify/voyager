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

#include <StreamUtils.h>
#include <jni.h>

class JavaInputStream : public InputStream {
public:
  // This input stream stores a temporary buffer to copy between Java and C++;
  // if we don't set a maximum buffer size here, the C++ side could read
  // hundreds of GB at once, which would allocate 2x that amount.
  static constexpr long long MAX_BUFFER_SIZE = 1024 * 1024 * 100;

  JavaInputStream(JNIEnv *env, jobject inputStream)
      : env(env), inputStream(inputStream) {

    jclass inputStreamClass = env->FindClass("java/io/InputStream");
    if (!inputStreamClass) {
      throw std::runtime_error("Native code failed to find InputStream class!");
    }

    if (!env->IsInstanceOf(inputStream, inputStreamClass)) {
      throw std::runtime_error(
          "Provided Java object is not a java.io.InputStream!");
    }
  };

  virtual bool isSeekable() { return false; }

  virtual long long getTotalLength() { return -1; }

  virtual long long read(char *buffer, long long bytesToRead) {
    jmethodID readMethod = env->GetMethodID(
        env->FindClass("java/io/InputStream"), "read", "([BII)I");

    if (!readMethod) {
      throw std::runtime_error("Native code failed to find "
                               "java.io.InputStream#read(byte[]) method!");
    }

    long long bytesRead = 0;

    long long bufferSize = std::min(MAX_BUFFER_SIZE, bytesToRead);
    jbyteArray byteArray = env->NewByteArray(bufferSize);
    if (!byteArray) {
      throw std::domain_error(
          "Failed to instantiate Java byte array of size: " +
          std::to_string(bufferSize));
    }

    while (bytesRead < bytesToRead) {
      int readResult = env->CallIntMethod(
          inputStream, readMethod, byteArray, 0,
          (int)(std::min(bufferSize, bytesToRead - bytesRead)));
      if (env->ExceptionCheck()) {
        return 0;
      }

      if (readResult > 0) {
        if (bytesRead + readResult > bytesToRead) {
          throw std::domain_error("java.io.InputStream#read(byte[]) returned " +
                                  std::to_string(readResult) + ", but only " +
                                  std::to_string(bytesToRead - bytesRead) +
                                  " bytes were required.");
        }

        if (readResult > bufferSize) {
          throw std::domain_error("java.io.InputStream#read(byte[]) returned " +
                                  std::to_string(readResult) +
                                  ", but buffer is only " +
                                  std::to_string(bufferSize) + " bytes.");
        }
        env->GetByteArrayRegion(byteArray, 0, readResult, (jbyte *)buffer);
        bytesRead += readResult;
        buffer += readResult;
      } else {
        bytesRead = readResult;
        break;
      }
    }

    env->DeleteLocalRef(byteArray);

    return bytesRead;
  }

  virtual bool isExhausted() { return false; }

  virtual long long getPosition() { return bytesRead; }

  virtual bool setPosition(long long position) { return false; }

  virtual ~JavaInputStream() {}

private:
  JNIEnv *env;
  jobject inputStream;
  long long bytesRead = 0;
};