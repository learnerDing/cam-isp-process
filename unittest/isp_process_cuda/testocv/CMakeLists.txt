cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

set(CMAKE_CXX_STANDARD 11)

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+simd -fopenmp")

# 手动设置 OpenMP 的 C++ 标志和库
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/usr/lib/aarch64-linux-gnu/libomp.so")  # 根据你的系统路径调整

# 查找 OpenMP
find_package(OpenMP REQUIRED)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(test test4.cpp)

# 链接 OpenMP 和 OpenCV
target_link_libraries(test
    PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)