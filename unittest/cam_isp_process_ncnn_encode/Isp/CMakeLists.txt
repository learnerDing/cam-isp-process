cmake_minimum_required(VERSION 3.10)
project(Isp LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenMP配置
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ARMv8 NEON优化配置
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+simd -mfpu=neon")
endif()

# 查找OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# 查找FFmpeg组件
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(SWSCALE REQUIRED libswscale)

# 源文件列表
set(ISP_SOURCES
    isp_opencv_pipeline.cpp
)

# 创建静态库
add_library(Isp STATIC ${ISP_SOURCES})

# 包含目录配置
target_include_directories(Isp PRIVATE
    ../include
    ${OpenCV_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
)

# 链接库配置
target_link_libraries(Isp
    PRIVATE
    Util
    ${OpenCV_LIBS}
    ${AVUTIL_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    OpenMP::OpenMP_CXX
)
