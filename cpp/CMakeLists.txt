cmake_minimum_required(VERSION 3.20)
project(Voyager)

set(CMAKE_CXX_STANDARD 17)
set(LLVM_CXX_STD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
