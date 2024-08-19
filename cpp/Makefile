default_target: build

BUILD_DIR := build

cmake:
	cmake -S . -B $(BUILD_DIR)

build: cmake
	cmake --build ${BUILD_DIR}

test: build
	cmake --build ${BUILD_DIR} --target test

clean:
	rm -rf ${BUILD_DIR}/*