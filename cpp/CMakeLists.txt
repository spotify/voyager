cmake_minimum_required(VERSION 3.20)
project(Voyager)

set(CMAKE_CXX_STANDARD 17)
set(LLVM_CXX_STD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

# Define the command
set(FIND_COMMAND find .. -type d -path ./include -prune -o -type d  -path ../python/.tox -prune -o -type f -name "*.cpp" -o -name "*.h"  -print)
set(CHECK_FORMAT_COMMAND clang-format --dry-run -i)
set(FORMAT_COMMAND clang-format --verbose -i)

# Check formatting only
add_custom_target(check-formatting
    COMMAND ${CHECK_FORMAT_COMMAND} `${FIND_COMMAND}`
    COMMENT "Checking C++ formatting"
)

# Run formatter
add_custom_target(format
    COMMAND ${FORMAT_COMMAND} `${FIND_COMMAND}`
    COMMENT "Running C++ formatter"
)
