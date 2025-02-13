cmake_minimum_required(VERSION 3.10)
project(Encode LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找FFmpeg组件
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVUTIL REQUIRED libavutil)

# 添加编码模块静态库
add_library(Encode STATIC
    EncodeThread.cpp
)

# 包含目录配置
target_include_directories(Encode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/Encode
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
)

# 链接依赖库
target_link_libraries(Encode PRIVATE
    Util              # 基础工具库
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
)

# # 安装规则（可选）
# install(TARGETS Encode
#     ARCHIVE DESTINATION lib
# )
# install(DIRECTORY ../include/Encode
#     DESTINATION include
# )
