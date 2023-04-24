# Makefile to build voyager's JNI bindings for macOS, Windows, or Linux.

.PHONY: all clean test

UNAME_S := $(shell uname -s)

MAC_SOBJ    := libvoyager.dylib
LINUX_SOBJ  := libvoyager.so
CPP_SRC_DIR := ../cpp/

CC        ?= clang++
SOURCE    := com_spotify_voyager_jni_Index.cpp
HEADERS   := $(SOURCE:.cpp=.h)
JAVA_SRC  := ./src/main/java

ifeq ($(UNAME_S),Linux)
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/linux
ALL_OBJS  := target/classes/linux-x64/$(LINUX_SOBJ) target/classes/linux-aarch64/$(LINUX_SOBJ)
CXXFLAGS  := -I. -lc -shared -std=c++17 -I ./include $(addprefix -I,$(JAVA_INC)) -I $(CPP_SRC_DIR) -fPIC -O3
else ifeq ($(UNAME_S),Darwin)
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/darwin
ALL_OBJS  := target/classes/mac-x64/$(MAC_SOBJ) target/classes/mac-aarch64/$(MAC_SOBJ)
CXXFLAGS  := -I. -lc++ -shared -std=c++17 -I ./include $(addprefix -I,$(JAVA_INC)) -I $(CPP_SRC_DIR) -fPIC -O3
else ifdef OS # Windows:
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/win32
ALL_OBJS  := target/classes/win-x64/$(MAC_SOBJ)
CXXFLAGS  := /I. /std:c++17 /I.\include $(addprefix -I,$(JAVA_INC)) /I $(CPP_SRC_DIR) /O2
endif

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
	$(CC) $(CXXFLAGS) -I linux-build/include -o $@ $(SOURCE)

target/classes/linux-aarch64/$(LINUX_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/linux-aarch64/
	mkdir -p linux-build
	cp -r $(CPP_SRC_DIR) linux-build/include
	cp -r $(addsuffix /*,$(JAVA_INC)) linux-build/include
	$(CC) $(CXXFLAGS) -I linux-build/include -o $@ $(SOURCE)

target/classes/mac-x64/$(MAC_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/mac-x64/
	$(CC) $(CXXFLAGS) -o $@ $(SOURCE) -arch x86_64

target/classes/mac-aarch64/$(MAC_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/mac-aarch64/
	$(CC) $(CXXFLAGS) -o $@ $(SOURCE) -arch arm64

test: $(ALL_OBJS)
	mvn test

clean:
	rm -rfv com_spotify_*.h classpath.txt target
