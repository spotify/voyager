add_executable(test main.cpp)
target_link_libraries(test
    PUBLIC
        VoyagerLib
    PRIVATE 
        Catch2::Catch2WithMain
)

target_compile_options(test PRIVATE -O2 -g)

include(CTest)
catch_discover_tests(test)
