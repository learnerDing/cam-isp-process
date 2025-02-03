#cmake_minimum_required(VERSION 2.6)
cmake_minimum_required(VERSION 3.1)  # 需要支持 C++11
project(MyYOLOv5)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖库
find_package(ncnn REQUIRED)      # 假设ncnn已安装或通过CMAKE_PREFIX_PATH指定
find_package(OpenCV REQUIRED     # 强制要求OpenCV存在
  COMPONENTS core highgui imgproc videoio imgcodecs)  # 明确指定需要的OpenCV组件

# 收集所有源码文件（自动包含当前目录及provider子目录）
file(GLOB_RECURSE SOURCES 
  "*.cpp"
  #"provider/*.cpp"
)

# 包含头文件路径
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}     # 当前目录头文件
  ${CMAKE_CURRENT_SOURCE_DIR}/provider  # provider子目录头文件
  ${ncnn_INCLUDE_DIRS}            # ncnn头文件路径
  ${OpenCV_INCLUDE_DIRS}          # OpenCV头文件路径
)

# 创建可执行文件
add_executable(my_yolov5 ${SOURCES})

# 链接库
target_link_libraries(my_yolov5 PRIVATE
  ncnn            # ncnn主库（假设已包含Vulkan支持）
  ${OpenCV_LIBS}  # OpenCV库
)