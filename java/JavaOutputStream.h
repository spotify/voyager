#include <StreamUtils.h>
#include <jni.h>

class JavaOutputStream : public OutputStream {
  static constexpr unsigned long long MAX_BUFFER_SIZE = 1024 * 1024 * 100;

public:
  JavaOutputStream(JNIEnv *env, jobject outputStream)
      : env(env), outputStream(outputStream) {
    jclass outputStreamClass = env->FindClass("java/io/OutputStream");
    if (!outputStreamClass) {
      throw std::runtime_error(
          "Native code failed to find OutputStream class!");
    }

    if (!env->IsInstanceOf(outputStream, outputStreamClass)) {
      throw std::runtime_error(
          "Provided Java object is not a java.io.OutputStream!");
    }
  };

  virtual void flush() {
    jmethodID flushMethod = env->GetMethodID(
        env->FindClass("java/io/OutputStream"), "flush", "()V");
    env->CallVoidMethod(outputStream, flushMethod);

    if (env->ExceptionCheck()) {
      throw std::runtime_error("JNI exception was thrown!");
    }
  }

  virtual bool write(const char *ptr, unsigned long long numBytes) {
    jmethodID writeMethod = env->GetMethodID(
        env->FindClass("java/io/OutputStream"), "write", "([B)V");

    if (!writeMethod) {
      throw std::runtime_error("Native code failed to find "
                               "java.io.OutputStream#write(byte[]) method!");
    }

    for (unsigned long long i = 0; i < numBytes; i += MAX_BUFFER_SIZE) {
      unsigned long long chunkSize = std::min(numBytes - i, MAX_BUFFER_SIZE);

      jbyteArray byteArray = env->NewByteArray(chunkSize);
      if (!byteArray) {
        throw std::domain_error(
            "Failed to instantiate Java byte array of size: " +
            std::to_string(chunkSize));
      }

      env->SetByteArrayRegion(byteArray, 0, chunkSize, (const jbyte *)ptr);
      env->CallVoidMethod(outputStream, writeMethod, byteArray);
      env->DeleteLocalRef(byteArray);

      if (env->ExceptionCheck()) {
        return false;
      }

      ptr += chunkSize;
    }

    return true;
  }

  virtual ~JavaOutputStream(){};

private:
  JNIEnv *env;
  jobject outputStream;
};