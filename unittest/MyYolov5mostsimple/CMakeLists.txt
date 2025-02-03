#cmake_minimum_required(VERSION 2.6)
cmake_minimum_required(VERSION 3.1)  # 需要支持 C++11
project(MyYOLOv5)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(OpenCV REQUIRED     # 强制要求OpenCV存在
  COMPONENTS core highgui imgproc videoio imgcodecs)  # 明确指定需要的OpenCV组件
# 查找线程库
find_package(Threads REQUIRED)  # 寻找thread库
find_package(OpenMP REQUIRED)

  # 手动指定ncnn路径
set(NCNN_INCLUDE_DIRS "/usr/local/include/ncnn")       # ncnn头文件路径
file(GLOB NCNN_LIBS  "/usr/local/lib/ncnn/libncnn.a"
                        "/usr/local/lib/ncnn/libglslang.a"
                      "/usr/local/lib/ncnn/libncnn.a"
                      "/usr/local/lib/ncnn/libSPIRV.a"
                      "/usr/local/lib/ncnn/libMachineIndependent.a"
                      "/usr/local/lib/ncnn/libOGLCompiler.a"
                      "/usr/local/lib/ncnn/libOSDependent.a"
                      "/usr/local/lib/ncnn/libGenericCodeGen.a"              
                      "/usr/lib/aarch64-linux-gnu/libvulkan.so")
set(NCNN_LIBRARIES ${NCNN_LIBS})  
# 收集所有源码文件（自动包含当前目录及provider子目录）
set(SOURCES
  main.cpp
  yolov5.cpp
)

# 包含头文件路径
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}     # 当前目录头文件
  # ${CMAKE_CURRENT_SOURCE_DIR}/provider  # provider子目录头文件
  ${NCNN_INCLUDE_DIRS}            # ncnn头文件路径
  ${OpenCV_INCLUDE_DIRS}          # OpenCV头文件路径
)
# 获取当前目标的INCLUDE_DIRECTORIES属性
get_property(include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

# 打印出包含的文件夹路径
message(STATUS "Include directories: ${include_dirs}")
add_link_options(-fopenmp -s)
# 创建可执行文件
add_executable(my_yolov5 ${SOURCES})

# 链接库
target_link_libraries(my_yolov5 PRIVATE
"-Wl,--start-group"
${NCNN_LIBRARIES}       # ncnn主库（假设已包含Vulkan支持）
"-Wl,--end-group"    
  ${OpenCV_LIBS}# OpenCV库
   -ldl -lpthread
)