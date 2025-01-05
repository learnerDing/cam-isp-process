cmake_minimum_required(VERSION 3.10)
project(EncodeVideoExample)

# 设置C标准
set(CMAKE_C_STANDARD 99)

# 查找FFmpeg库
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libavdevice libavfilter libswresample)

# 添加可执行文件
add_executable(yuv_encode yuv_encode.c)

# 包含FFmpeg头文件
target_include_directories(yuv_encode PRIVATE ${FFMPEG_INCLUDE_DIRS})

# 链接FFmpeg库
target_link_libraries(yuv_encode PRIVATE ${FFMPEG_LIBRARIES})

# 添加数学库（如果需要）
target_link_libraries(yuv_encode PRIVATE m)
# 显式链接 pthread 库
target_link_libraries(yuv_encode PRIVATE pthread)