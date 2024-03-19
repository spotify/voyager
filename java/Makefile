# Makefile to build voyager's JNI bindings for macOS, Windows, or Linux.

.PHONY: all clean test

UNAME_S := $(shell uname -s)

MAC_SOBJ    := libvoyager.dylib
LINUX_SOBJ  := libvoyager.so
WIN_SOBJ    := voyager.dll
CPP_SRC_DIR := ../cpp/

SOURCE    := com_spotify_voyager_jni_Index.cpp
HEADERS   := $(SOURCE:.cpp=.h)
JAVA_SRC  := ./src/main/java

ifeq ($(UNAME_S),Linux)
CXX       ?= g++
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/linux
ALL_OBJS  := target/classes/linux-x64/$(LINUX_SOBJ) target/classes/linux-aarch64/$(LINUX_SOBJ)
CXXFLAGS  := -I. -lc -shared -std=c++17 -I ./include $(addprefix -I,$(JAVA_INC)) -I $(CPP_SRC_DIR) -fPIC -O3
PREBUILD_COMMAND := sudo apt-get update && sudo apt-get install -y ca-certificates-java openjdk-11-jre-headless; { curl https://dlcdn.apache.org/maven/maven-3/3.9.2/binaries/apache-maven-3.9.2-bin.tar.gz | sudo tar -xvzf - -C /opt ; }; export M2_HOME=/opt/apache-maven-3.9.2 && export PATH=\"${M2_HOME}/bin:${PATH}\"
else ifeq ($(UNAME_S),Darwin)
CXX       ?= clang++
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/darwin
ALL_OBJS  := target/classes/mac-x64/$(MAC_SOBJ) target/classes/mac-aarch64/$(MAC_SOBJ)
CXXFLAGS  := -I. -lc++ -shared -std=c++17 -I ./include $(addprefix -I,$(JAVA_INC)) -I $(CPP_SRC_DIR) -fPIC -O3
else ifdef OS # Windows:
CXX       := cl.exe
JAVA_INC  := $(JAVA_HOME)/include $(JAVA_HOME)/include/win32
ALL_OBJS  := target/classes/win-x64/$(WIN_SOBJ)
CXXFLAGS  := /I. /EHsc /DLL /LD /std:c++17 /I.\include $(addprefix -I,$(JAVA_INC)) /I $(CPP_SRC_DIR) /O2
endif

# Allow linking against ASan if USE_ASAN is set to 1:
ifeq (${USE_ASAN},1) 
CXXFLAGS := $(CXXFLAGS) -fsanitize=address
else
endif


all: $(ALL_OBJS)
	# Done!

check-java-home:
ifndef JAVA_HOME
	$(error JAVA_HOME is undefined)
endif

classpath.txt: check-java-home
	mvn dependency:build-classpath | grep '\.jar[;:]' | tr ';' ':' > $@

%.h: %.cpp classpath.txt
	# For each Java JNI .cpp file, we need a corresponding header to be generated.
	$(eval JAVA_FILE := $(shell python3 -c 'print("$<".split(".")[0].replace("_", "/") + ".java")'))
	javac -cp ./target/classes:$(shell cat classpath.txt) -h . -Xlint:deprecation -sourcepath $(JAVA_SRC) $(JAVA_SRC)/$(JAVA_FILE)

target/classes/linux-x64/$(LINUX_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/linux-x64/
	mkdir -p linux-build
	cp -r $(CPP_SRC_DIR) linux-build/include
	cp -r $(addsuffix /*,$(JAVA_INC)) linux-build/include
	# Why use Dockcross if we're already building on x86?
	# We need to use an older version of GLIBC and GLIBCXX to ensure wide compatibility on Linux.
	docker run -v $(shell realpath ../):/work dockcross/linux-x64:20230202-378e837 bash -x -c 'cd /work/java && $(PREBUILD_COMMAND) && $$CXX $(CXXFLAGS) -I linux-build/include -o $@ $(SOURCE)'

target/classes/linux-aarch64/$(LINUX_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/linux-aarch64/
	mkdir -p linux-build
	cp -r $(CPP_SRC_DIR) linux-build/include
	cp -r $(addsuffix /*,$(JAVA_INC)) linux-build/include
	docker run -v $(shell realpath ../):/work dockcross/linux-arm64-lts bash -x -c 'cd /work/java && $(PREBUILD_COMMAND) && $$CXX $(CXXFLAGS) -I linux-build/include -o $@ $(SOURCE)'

target/classes/mac-x64/$(MAC_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/mac-x64/
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCE) -arch x86_64

target/classes/mac-aarch64/$(MAC_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/mac-aarch64/
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCE) -arch arm64

target/classes/win-x64/$(WIN_SOBJ): classpath.txt $(HEADERS)
	mkdir -p target/classes/win-x64/
	$(CXX) $(CXXFLAGS) $(SOURCE)
	cp com_spotify_voyager_jni_Index.dll $@

test: $(ALL_OBJS)
	mvn test

clean:
	rm -rfv com_spotify_*.h classpath.txt target
