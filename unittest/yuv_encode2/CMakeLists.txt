cmake_minimum_required(VERSION 3.10)
project(EncodeVideoExample)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 FFmpeg 库
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice libavfilter libswresample)

# 添加源文件
set(SOURCES
    main.cpp
    FrameQueue.cpp
    EncodeThread.cpp
    Thread.cpp
)

# 添加可执行文件
add_executable(yuv_encode ${SOURCES})

# 包含 FFmpeg 头文件
target_include_directories(yuv_encode PRIVATE ${FFMPEG_INCLUDE_DIRS})

# 链接 FFmpeg 库
target_link_libraries(yuv_encode PRIVATE ${FFMPEG_LIBRARIES})

# 添加数学库（如果需要）
target_link_libraries(yuv_encode PRIVATE m)

# 显式链接 pthread 库
target_link_libraries(yuv_encode PRIVATE pthread)