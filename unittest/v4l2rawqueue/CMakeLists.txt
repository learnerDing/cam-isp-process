cmake_minimum_required(VERSION 3.10)
project(v4l2rawqueue)

# 设置 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 添加可执行文件
add_executable(v4l2rawqueue
    src/Cam/Cam.cpp
    src/Cam/CamThread.cpp
    src/Cam/v4l2cam.cpp
    utils/FrameQueue.cpp
    utils/Thread.cpp
    main.cpp
)

# 链接 OpenCV 库
target_link_libraries(v4l2rawqueue ${OpenCV_LIBS})