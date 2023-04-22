# Makefile to build voyager's JNI bindings for macOS or Linux, from a macOS machine.
# Requires a working Docker and JVM installation.
# For Apple Silicon support, use `make target/classes/mac-aarch64/libvoyager.dylib`

.PHONY: all clean

UNAME_S := $(shell uname -s)

SOBJ        := libvoyager.dylib
LINUX_SOBJ  := libvoyager.so
CPP_SRC_DIR := ../cpp/

ifeq ($(UNAME_S),Linux)
# JAVA_HOME ?= $(shell echo ${JAVA_HOME})
JAVA_HOME ?= /usr/lib/jvm/java-11-openjdk-arm64/
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/linux
# Nothing, as we're already on Linux:
LINUX_DOCKER_COMMAND := 
ALL_OBJS  := target/classes/linux-x64/$(LINUX_SOBJ) target/classes/linux-aarch64/$(LINUX_SOBJ)
else
JAVA_HOME ?= $(shell /usr/libexec/java_home)
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/darwin
LINUX_DOCKER_COMMAND := docker run -v $(shell pwd):/work dockcross/manylinux2014-x64
ALL_OBJS  := target/classes/mac-x64/$(SOBJ) target/classes/mac-aarch64/$(SOBJ) target/classes/linux-x64/$(LINUX_SOBJ) target/classes/linux-aarch64/$(LINUX_SOBJ)
endif

CC        := g++
CXXFLAGS  := -I. -lc -shared -std=c++17
JNIFLAGS  := -I ./include $(addprefix -I,$(JAVA_INC)) -I $(CPP_SRC_DIR)
FASTFLAGS := -fPIC -O3
SOURCE    := com_spotify_voyager_jni_Index.cpp
HEADERS   := $(SOURCE:.cpp=.h)
JAVA_SRC  := ./src/main/java

all: $(ALL_OBJS)
	# Done!

classpath.txt:
	mvn dependency:build-classpath | grep '\.jar:' > $@

%.h: %.cpp classpath.txt
	# For each Java JNI .cpp file, we need a corresponding header to be generated.
	$(eval JAVA_FILE := $(shell python3 -c 'print("$<".split(".")[0].replace("_", "/") + ".java")'))
	javac -cp ./target/classes:$(shell cat classpath.txt) -h . -sourcepath $(JAVA_SRC) $(JAVA_SRC)/$(JAVA_FILE)

target/classes/linux-x64/$(LINUX_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/linux-x64/
	mkdir -p linux-build
	cp -r $(CPP_SRC_DIR) linux-build/include
	cp -r $(addsuffix /*,$(JAVA_INC)) linux-build/include
	$(LINUX_DOCKER_COMMAND) $(CC) $(CXXFLAGS) -I linux-build/include $(FASTFLAGS) -o $@ $(SOURCE) $(DEBUGFLAGS)
	# rm -rf linux-build

target/classes/linux-aarch64/$(LINUX_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/linux-aarch64/
	mkdir -p linux-build
	cp -r $(CPP_SRC_DIR) linux-build/include
	cp -r $(addsuffix /*,$(JAVA_INC)) linux-build/include
# docker run -v $(shell pwd):/work --entrypoint bash ubuntu:18.04 -c 'apt-get update && apt-get install -y build-essential default-jdk && cd /work && 
	$(CC) $(CXXFLAGS) -I linux-build/include $(FASTFLAGS) -o $@ $(SOURCE) $(DEBUGFLAGS)
# '
	# rm -rf linux-build

target/classes/mac-x64/$(SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/mac-x64/
	$(CC) $(CXXFLAGS) $(JNIFLAGS) $(FASTFLAGS) -o $@ $(SOURCE) $(DEBUGFLAGS) -arch x86_64

target/classes/mac-aarch64/$(SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/mac-aarch64/
	$(CC) $(CXXFLAGS) $(JNIFLAGS) $(FASTFLAGS) -o $@ $(SOURCE) $(DEBUGFLAGS) -arch arm64

test: $(ALL_OBJS)
	mvn test

release: target/classes/mac-x64/$(SOBJ) target/classes/mac-aarch64/$(SOBJ) target/classes/linux-x64/$(LINUX_SOBJ)
	mvn -B release:clean release:prepare release:perform -Dmaven.wagon.http.retryHandler.count=20

clean:
	rm -rfv com_spotify_*.h classpath.txt target
