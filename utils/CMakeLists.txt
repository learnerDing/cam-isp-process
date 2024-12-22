add_library(convert convert.cpp)

add_library(Pipeline
    pipeline.cpp 
    macro.h 
    string_utils.h 
    Tensor.h 
)
target_link_libraries(Pipeline PUBLIC convert)
