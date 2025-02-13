cmake_minimum_required(VERSION 3.10)
project(Util LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加所有基础源文件
file(GLOB UTIL_SOURCES 
    "*.cpp"
)

# 编译静态库
add_library(Util STATIC ${UTIL_SOURCES})

# 设置公共头文件目录（其他模块通过target_link_libraries自动继承）
target_include_directories(Util PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/util # 公共头文件暴露给链接者
)

# # 安装规则（可选）
# install(TARGETS Util
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
# )
# install(DIRECTORY ../include/util
#     DESTINATION include
# )