# isp_process_ocv/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(isp_process_ocv)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+simd -fopenmp")

# 手动设置OpenMP支持
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/usr/lib/aarch64-linux-gnu/libomp.so")  # 根据你的系统路径调整
find_package(OpenMP REQUIRED)

# 查找其他依赖项
find_package(OpenCV REQUIRED COMPONENTS core highgui)
find_package(Threads REQUIRED)

# 设置包含目录
include_directories(
    include
    rawprovide
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 添加子目录
add_subdirectory(rawprovide)

# 添加主程序可执行文件
add_executable(${PROJECT_NAME}
    main.cpp
    src/isp_opencv_pipeline.cpp
    src/FrameQueue.cpp
    src/Thread.cpp
)

# 链接依赖项
target_link_libraries(${PROJECT_NAME} PRIVATE
    Raw8Provider
    ${OpenCV_LIBS}
    Threads::Threads
    OpenMP::OpenMP_CXX  # 显式链接OpenMP库
)

# 添加针对ARM NEON的优化选项
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    target_compile_options(${PROJECT_NAME} PRIVATE -march=armv8-a+simd)
endif()
