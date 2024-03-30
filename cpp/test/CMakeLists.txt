add_executable(test test_main.cpp)


target_link_libraries(test
    PUBLIC
        VoyagerLib
    PRIVATE 
        doctest
)

target_compile_options(test PRIVATE -O2 -g)

include(CTest)
