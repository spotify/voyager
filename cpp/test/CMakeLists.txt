# List the test source files
set(TEST_FILES test_main.cpp doctest_setup.cpp)  # Add any test files here

# Create an executable for the tests
add_executable(VoyagerTests ${TEST_FILES})

# Add compiler flags
target_compile_options(VoyagerTests PRIVATE -g)

# Link the test executable with the main project and Doctest
# target_link_libraries(MyProjectTests PRIVATE MyProject doctest::doctest)
target_link_libraries(VoyagerTests
    PUBLIC
        VoyagerLib
    PRIVATE 
        doctest
)

# Discover tests using Doctest
include(${CMAKE_SOURCE_DIR}/include/doctest/scripts/cmake/doctest.cmake)
doctest_discover_tests(VoyagerTests)
