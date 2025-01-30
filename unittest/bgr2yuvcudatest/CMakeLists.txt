cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(openmp_thread_neon_cuda_bgr2yuv LANGUAGES CXX CUDA)

# 查找 CUDA
find_package(CUDA 10.0 REQUIRED)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
find_package(CUDA REQUIRED)

# 设置编译器标志
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

# 设置 CUDA 架构
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      ")
set(CMAKE_CUDA_ARCHITECTURES 70 75 )
message("-- Assign GPU architecture (sm=70 75)")

# 设置调试模式标志
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")


# 添加可执行目标
add_executable(${PROJECT_NAME}
    bgr2yuvtest.cpp
    bgr2yuv.cu
)

# 包含头文件目录
target_include_directories(${PROJECT_NAME} PRIVATE include  ${CUDA_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES} )

# 配置 C++ 标准
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
)