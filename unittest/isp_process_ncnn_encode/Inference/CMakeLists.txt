cmake_minimum_required(VERSION 3.10)
project(Inference)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加预处理器定义
add_definitions(-DYOLOV5_V62)  # 根据代码需要定义版本宏

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)    # 确保ncnnConfig.cmake路径已正确配置

# 设置头文件包含路径
include_directories(
    ../../include              # 项目公共头文件
    ${OpenCV_INCLUDE_DIRS}     # OpenCV头文件
    ${ncnn_INCLUDE_DIRS}       # ncnn头文件
)

# 设置源文件
set(SOURCES
    InferenceThread.cpp
    yolov5.cpp
)

# 创建静态库
add_library(Inference STATIC ${SOURCES})

# 添加链接库
target_link_libraries(Inference
    PRIVATE
    Util
    ${OpenCV_LIBS}
    ncnn
)

# 针对ARMv8的优化选项
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(Inference PRIVATE -march=armv8-a+simd)
endif()

# 安装模型文件（可选）
configure_file(yolov5s_6.2.bin ${CMAKE_CURRENT_BINARY_DIR}/yolov5s_6.2.bin COPYONLY)
configure_file(yolov5s_6.2.param ${CMAKE_CURRENT_BINARY_DIR}/yolov5s_6.2.param COPYONLY)
