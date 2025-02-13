# Cam/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(Cam LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARMv8 NEON优化配置
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a+simd -fopenmp")
endif()

# 查找依赖项目
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# 添加库目标
add_library(Cam STATIC
    Cam.cpp
    v4l2cam.cpp
)

# 包含目录
target_include_directories(Cam
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include/Cam  # 假设头文件在项目根目录的include/Cam下
    PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)

# 链接依赖库
target_link_libraries(Cam
    PRIVATE
        Util  # 假设util模块生成Util库
    PUBLIC
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)

# 安装规则（可选）
# install(TARGETS Cam ARCHIVE DESTINATION lib)
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION include/Cam FILES_MATCHING PATTERN "*.h")
