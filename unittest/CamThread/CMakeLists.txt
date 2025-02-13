cmake_minimum_required(VERSION 3.10)
project(CamTest LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARMv8 NEON优化配置
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+simd -fopenmp")
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# 手动设置 OpenMP 的 C++ 标志和库
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/usr/lib/aarch64-linux-gnu/libomp.so")  # 根据你的系统路径调整
# 查找OpenMP
find_package(OpenMP REQUIRED)

# 添加可执行文件
add_executable(CamTest
    Camtest.cpp
    util/FrameQueue.cpp
    util/Thread.cpp
    Cam.cpp
    v4l2cam.cpp
)
target_include_directories(CamTest PRIVATE  ${OpenCV_INCLUDE_DIRS}
                                            ${CMAKE_SOURCE_DIR}
                                            ${CMAKE_SOURCE_DIR}/util)
# 链接库
target_link_libraries(CamTest
    PRIVATE
    # OpenCV::OpenCV #使用新链接方式，就不需要在手动包含opencv的头文件目录了
    ${OpenCV_LIBS} 
    OpenMP::OpenMP_CXX
)

# # 安装规则（可选）
# install(TARGETS CamTest DESTINATION bin)
