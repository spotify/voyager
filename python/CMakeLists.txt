cmake_minimum_required(VERSION 3.15...3.27)
project(voyager LANGUAGES CXX)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

set(IS_UNIX ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
set(IS_WINDOWS ${CMAKE_SYSTEM_NAME} STREQUAL "Windows")

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE})

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Allow for some math optimization on unix but not -ffast-math
# See: https://simonbyrne.github.io/notes/fastmath/#flushing_subnormals_to_zero
if (IS_UNIX)
  add_compile_options(-fassociative-math -fno-signaling-nans -fno-trapping-math -fno-signed-zeros -freciprocal-math -fno-math-errno)
endif()

# Relase build with all optimizations
if (CMAKE_BUILD_TYPE STREQUAL "Release" AND IS_UNIX)
  add_compile_options(-O3)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release" AND IS_WINDOWS)
  add_compile_options(/Ox)
endif()


# Check the USE_ASAN environment variable, and if set to 1, enable AddressSanitizer
if (DEFINED ENV{USE_ASAN} AND "$ENV{USE_ASAN}" STREQUAL "1")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()


nanobind_add_module(
  voyager_ext
  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  NB_STATIC STABLE_ABI LTO FREE_THREADED NOMINSIZE
  # Sources:
  src/bindings.cpp
)

# Install directive for scikit-build-core
install(TARGETS voyager_ext LIBRARY DESTINATION voyager)
