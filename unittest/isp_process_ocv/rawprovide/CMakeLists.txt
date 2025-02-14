#isp_process_ocv/rawprovide/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(rawprovider_exe)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖项
find_package(OpenCV REQUIRED COMPONENTS core highgui)
find_package(Threads REQUIRED)


# 创建静态库
add_library(Raw8Provider STATIC
    Raw8Provider.cpp
    # ../src/FrameQueue.cpp       # 假设FrameQueue.cpp在父目录
    # ../src/Thread.cpp           # 假设Thread.cpp在父目录
)
#头文件链接
target_include_directories(Raw8Provider PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接库依赖项
target_link_libraries(Raw8Provider PRIVATE 
    ${OpenCV_LIBS}
    Threads::Threads
)

