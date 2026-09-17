add_library(VoyagerLib cpu_utils.cpp)
target_include_directories(VoyagerLib 
    PUBLIC
        .
        Spaces
)
