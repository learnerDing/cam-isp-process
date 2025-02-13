cmake_minimum_required(VERSION 3.10)
project(rawprovider_exe)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖项
find_package(OpenCV REQUIRED COMPONENTS core highgui)
find_package(Threads REQUIRED)

# 添加头文件目录
include_directories(
    ../../include            # 项目公共头文件
    ${CMAKE_CURRENT_SOURCE_DIR}  # Raw8Provider.h所在目录
)

# 创建静态库
add_library(Raw8Provider STATIC
    Raw8Provider.cpp
    ../FrameQueue.cpp       # 假设FrameQueue.cpp在父目录
    ../Thread.cpp           # 假设Thread.cpp在父目录
)

# 链接库依赖项
target_link_libraries(Raw8Provider PRIVATE 
    ${OpenCV_LIBS}
    Threads::Threads
)

# 创建可执行文件
add_executable(${PROJECT_NAME} main.cpp)

# 链接可执行文件到库
target_link_libraries(${PROJECT_NAME} PRIVATE
    Raw8Provider
    ${OpenCV_LIBS}
)
