cmake_minimum_required(VERSION 3.8)
project(bgr2yuv_benchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenMP
find_package(OpenMP REQUIRED)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# ARM平台NEON优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        #add_compile_options(-mfpu=neon -march=armv7-a -O3)
        add_compile_options(-march=armv8-a -O3)
    endif()
endif()

add_executable(bgr2yuv_benchmark
    main.cpp
)

# 链接OpenMP和OpenCV
target_link_libraries(bgr2yuv_benchmark
    PRIVATE
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)