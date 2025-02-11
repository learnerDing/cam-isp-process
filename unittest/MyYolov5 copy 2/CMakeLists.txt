cmake_minimum_required(VERSION 3.1)
project(MyYOLOv5)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 新增：启用线程支持
set(THREADS_PREFER_PTHREAD_FLAG ON)

# 查找依赖库
find_package(OpenCV REQUIRED
  COMPONENTS core highgui imgproc videoio imgcodecs)
find_package(Threads REQUIRED)  # 新增：显式查找pthread
find_package(OpenMP REQUIRED)

# 手动指定ncnn路径（保持原样）
set(NCNN_INCLUDE_DIRS "/usr/local/include/ncnn")
file(GLOB NCNN_LIBS  
  "/usr/local/lib/ncnn/libncnn.a"
  "/usr/local/lib/ncnn/libglslang.a"
  "/usr/local/lib/ncnn/libSPIRV.a"
  "/usr/local/lib/ncnn/libMachineIndependent.a"
  "/usr/local/lib/ncnn/libOGLCompiler.a"
  "/usr/local/lib/ncnn/libOSDependent.a"
  "/usr/local/lib/ncnn/libGenericCodeGen.a"              
  "/usr/lib/aarch64-linux-gnu/libvulkan.so")
set(NCNN_LIBRARIES ${NCNN_LIBS})

# 收集所有源码文件（新增多线程相关文件）
set(SOURCES
  main.cpp
  yolov5.cpp
  FrameQueue.cpp     # 新增
  InferenceThread.cpp  # 新增
  Thread.cpp         # 新增
)

# 包含头文件路径（新增include目录）
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include  # 新增头文件目录
  ${NCNN_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 保持原有的调试信息
get_property(include_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(STATUS "Include directories: ${include_dirs}")

add_link_options(-fopenmp -s)

# 创建可执行文件（保持原样）
add_executable(my_yolov5 ${SOURCES})

# 链接库（保持原有顺序，仅添加线程库）
target_link_libraries(my_yolov5 PRIVATE
  "-Wl,--start-group"
  ${NCNN_LIBRARIES}
  "-Wl,--end-group"    
  ${OpenCV_LIBS}
  Threads::Threads  # 显式链接pthread
  -ldl 
)