default_target: build

BUILD_DIR := build

cmake:
	cmake -S . -B $(BUILD_DIR)

build: cmake
	cmake --build ${BUILD_DIR}

test: build
	ctest --test-dir ${BUILD_DIR}

clean:
	rm -rf ${BUILD_DIR}/*