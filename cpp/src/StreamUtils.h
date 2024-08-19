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

#pragma once
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>

/**
 * Like std::istream, but custom with fewer methods to implement.
 */
class InputStream {
public:
  virtual ~InputStream() = default;
  virtual bool isSeekable() = 0;
  virtual long long getTotalLength() = 0;
  virtual long long read(char *buffer, long long bytesToRead) = 0;
  virtual bool isExhausted() = 0;
  virtual long long getPosition() = 0;
  virtual bool setPosition(long long position) = 0;
  virtual bool advanceBy(long long numBytes) { return setPosition(getPosition() + numBytes); }
  virtual uint32_t peek() = 0;
};

class FileInputStream : public InputStream {
public:
  FileInputStream(const std::string &filename) : filename(filename) {
    handle = fopen(filename.c_str(), "r");
    if (!handle) {
      throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    struct stat st;
    fstat(fileno(handle), &st);
    isRegularFile = (st.st_mode & S_IFMT) == S_IFREG;
    if (isRegularFile) {
      sizeInBytes = st.st_size;
    }
  }

  virtual bool isSeekable() { return isRegularFile; }
  virtual long long getTotalLength() { return sizeInBytes; }

  virtual long long read(char *buffer, long long bytesToRead) { return fread(buffer, 1, bytesToRead, handle); }

  virtual bool isExhausted() { return feof(handle); }
  virtual long long getPosition() { return ftell(handle); }
  virtual bool setPosition(long long position) { return fseek(handle, position, SEEK_SET) == 0; }
  virtual bool advanceBy(long long bytes) { return fseek(handle, bytes, SEEK_CUR) == 0; }
  virtual uint32_t peek() {
    uint32_t result = 0;
    long long lastPosition = getPosition();
    if (read((char *)&result, sizeof(result)) == sizeof(result)) {
      setPosition(lastPosition);
      return result;
    } else {
      throw std::runtime_error("Failed to peek " + std::to_string(sizeof(result)) + " bytes from file \"" + filename +
                               "\" at index " + std::to_string(lastPosition) + ".");
    }
  }

  virtual ~FileInputStream() {
    if (handle) {
      fclose(handle);
      handle = nullptr;
    }
  }

protected:
  FileInputStream() {}
  FILE *handle = nullptr;
  std::string filename;

private:
  bool isRegularFile = false;
  long long sizeInBytes = -1;
};

/**
 * Like std::ostream, but custom with fewer methods to implement.
 */
class OutputStream {
public:
  virtual ~OutputStream() = default;
  virtual void flush() = 0;
  virtual bool write(const char *ptr, unsigned long long numBytes) = 0;
};

class FileOutputStream : public OutputStream {
public:
  FileOutputStream(const std::string &filename) {
    errno = 0;
    handle = fopen(filename.c_str(), "wb");
    if (!handle) {
      throw std::runtime_error("Failed to open file for writing (errno " + std::to_string(errno) + "): " + filename);
    }
  }

  virtual bool write(const char *buffer, unsigned long long numBytes) {
    return fwrite((const void *)buffer, 1, numBytes, handle) == numBytes;
  }

  virtual void flush() { fflush(handle); }

  virtual ~FileOutputStream() {
    if (handle) {
      fclose(handle);
    }
  }

private:
  FILE *handle = nullptr;
};

class MemoryOutputStream : public OutputStream {
public:
  MemoryOutputStream() {}

  virtual bool write(const char *buffer, unsigned long long numBytes) {
    outputStream.write(buffer, numBytes);
    return true;
  }

  virtual void flush() {}

  std::string getValue() { return outputStream.str(); }

private:
  std::ostringstream outputStream;
};

template <typename T> static void writeBinaryPOD(std::shared_ptr<OutputStream> out, const T &podRef) {
  if (!out->write((char *)&podRef, sizeof(T))) {
    throw std::runtime_error("Failed to write " + std::to_string(sizeof(T)) + " bytes to stream!");
  }
}

template <typename T> static void readBinaryPOD(std::shared_ptr<InputStream> in, T &podRef) {
  long long bytesRead = in->read((char *)&podRef, sizeof(T));
  if (bytesRead != sizeof(T)) {
    throw std::runtime_error("Failed to read " + std::to_string(sizeof(T)) + " bytes from stream! Got " +
                             std::to_string(bytesRead) + ".");
  }
}
