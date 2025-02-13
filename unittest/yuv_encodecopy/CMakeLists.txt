#cmake_minimum_required(VERSION 2.6)
cmake_minimum_required(VERSION 3.1)  # 需要支持 C++11
project(yuv_encode)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS" )
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
# 添加源文件
set(SOURCES
    yuv_encode.cpp
    FrameQueue.cpp
    EncodeThread.cpp
    Thread.cpp
)
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
find_library(AVDEVICE_LIBRARY avdevice)
# 查找 OpenCV 库
find_package(OpenCV REQUIRED)
# 添加可执行文件
add_executable(yuv_encode ${SOURCES})

target_include_directories(yuv_encode PRIVATE ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
target_include_directories(yuv_encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
#target_link_libraries(yuv_encode PRIVATE ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY} -lpthread)
target_link_libraries(yuv_encode PRIVATE ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY} ${OpenCV_LIBS} -lpthread )