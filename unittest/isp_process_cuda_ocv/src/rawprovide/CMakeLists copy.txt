cmake_minimum_required(VERSION 3.10)
project(rawprovider_exe)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件目录（指向项目根目录的include）
include_directories(../../include)
get_property( dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES )

foreach( dir ${dirs} )
    message( STATUS "include dir=${dir}" )
endforeach()
# 查找OpenCV包
find_package(OpenCV REQUIRED COMPONENTS core highgui)

# 查找线程库
find_package(Threads REQUIRED)

# 定义可执行文件
add_executable(${PROJECT_NAME}
    main.cpp
    Raw8Provider.cpp
    ../FrameQueue.cpp
    ../Thread.cpp
)

# 链接依赖库
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)
