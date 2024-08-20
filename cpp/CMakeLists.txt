cmake_minimum_required(VERSION 3.20)
project(Voyager)

set(CMAKE_CXX_STANDARD 17)
set(LLVM_CXX_STD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

# Define our find command with any appropriate directory exclusions (add another with `-o -path <PATH> -prune`)
set(FIND_COMMAND find .. -path ../cpp/include -prune -o -path ../cpp/CMakeFiles -prune -o -path ../python/.tox -prune -o -name "*.cpp" -print -o -name "*.h" -type f -print)
set(CHECK_FORMAT_COMMAND clang-format --verbose --dry-run -i)
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
