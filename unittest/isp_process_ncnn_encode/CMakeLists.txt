cmake_minimum_required(VERSION 3.10)
project(isp_process_ncnn_encode LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 包含头文件目录
include_directories(include)

# 查找依赖包
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(ncnn REQUIRED)  # 假设你已经正确配置ncnn
find_package(FFmpeg REQUIRED) # 编码相关依赖

# 添加子目录（这些子目录会生成静态库）注意顺序
add_subdirectory(util)
add_subdirectory(Isp)
add_subdirectory(Inference)
add_subdirectory(Encode)
add_subdirectory(rawprovide)


# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp)

# 链接依赖库
target_link_libraries(${PROJECT_NAME} PRIVATE
    util
    Isp
    Inference
    Encode
    rawprovide
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ncnn
    ${FFmpeg_LIBRARIES}
    # 其他可能的依赖...
)

# # 添加CUDA支持（如果需要）
# if(CUDA_FOUND)
#     enable_language(CUDA)
#     target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
# endif()

# 设置输出目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
