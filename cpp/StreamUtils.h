#pragma once
#include <exception>
#include <iostream>
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
  virtual bool advanceBy(long long numBytes) {
    return setPosition(getPosition() + numBytes);
  }
};

class FileInputStream : public InputStream {
public:
  FileInputStream(const std::string &filename) {
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

  virtual long long read(char *buffer, long long bytesToRead) {
    return fread(buffer, 1, bytesToRead, handle);
  }

  virtual bool isExhausted() { return feof(handle); }
  virtual long long getPosition() { return ftell(handle); }
  virtual bool setPosition(long long position) {
    return fseek(handle, position, SEEK_SET) == 0;
  }
  virtual bool advanceBy(long long bytes) {
    return fseek(handle, bytes, SEEK_CUR) == 0;
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

private:
  bool isRegularFile = false;
  long long sizeInBytes = -1;
};

/**
 * An input stream that wraps a subprocess command.
 * Useful when loading a Voyager index from a remote filesystem
 * or other object storage directly into memory.
 */
class SubprocessInputStream : public FileInputStream {
public:
  SubprocessInputStream(const std::string &subprocessCommand) {
    handle =
#ifdef _MSC_VER
        _popen
#else
        popen
#endif
        (subprocessCommand.c_str(), "r");

    if (!handle) {
      throw std::runtime_error("Failed to open subprocess: " +
                               subprocessCommand);
    }
  }

  virtual bool isSeekable() { return false; }
  virtual long long getTotalLength() { return -1; }

  virtual ~SubprocessInputStream() {
    if (handle) {
#ifdef _MSC_VER
      _pclose
#else
      pclose
#endif
          (handle);

      handle = nullptr;
    }
  }
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
    handle = fopen(filename.c_str(), "w");
    if (!handle) {
      throw std::runtime_error("Failed to open file for writing (errno " +
                               std::to_string(errno) + "): " + filename);
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