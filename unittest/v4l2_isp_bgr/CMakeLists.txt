cmake_minimum_required(VERSION 3.10)
project(RawCvPro LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA 10.0 REQUIRED)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})


list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)
find_package(CUDA REQUIRED)

# setting compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")	
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -std=c++11")#打开c++调试
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        ")
#                      -rdc=true") # not sure the effect of this option, retain it temply

set(CMAKE_CUDA_ARCHITECTURES 70 75)
message("-- Assign GPU architecture (sm=70 75)")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_STANDARD STREQUAL "11")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

include_directories(${CMAKE_SOURCE_DIR}/../../src) #添加各种头文件寻找路径，编译器才找得到
include_directories(${CMAKE_SOURCE_DIR}/../../src/Kernel)
include_directories(${CMAKE_SOURCE_DIR}/../../src/layer)
include_directories(${CMAKE_SOURCE_DIR}/../../src/layer/camera)
include_directories(${CMAKE_SOURCE_DIR}/../../src/layer/processing)
include_directories(${CMAKE_SOURCE_DIR}/../../src/utils)

include_directories(${CMAKE_SOURCE_DIR}/)
get_property(dirs DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
message(">>> include_dirs=${dirs}") #打印一下目录情况

# 递归地查找所有 .cpp 文件
file(GLOB_RECURSE RawCVPro_CXX_SOURCES ${PROJECT_SOURCE_DIR}/../../src/*.cpp)

# 查找当前目录下的 .cpp 文件并追加到已存在的列表
file(GLOB RawCVPro_CXX_SOURCES_APPEND ./*.cpp) # 使用一个新的变量

# 将当前目录下的文件追加到原来的文件列表中
list(APPEND RawCVPro_CXX_SOURCES ${RawCVPro_CXX_SOURCES_APPEND})

# 打印文件列表
message(STATUS "RawCVPro_CXX_SOURCES:")
foreach(source ${RawCVPro_CXX_SOURCES})
    message(STATUS "  - ${source}")
endforeach()
file(GLOB_RECURSE RawCVPro_CUDA_SOURCES ${PROJECT_SOURCE_DIR}/../../src/*.cu)#把cu文件添加到变量RawPro_CUDA_SOURCES
message(STATUS "RawCVPro_CUDA_SOURCES:")
foreach(source ${RawCVPro_CUDA_SOURCES})
    message(STATUS "  - ${source}")
endforeach()
#尝试将cpp&cuda文件编译成静态库来运行
add_library(MyLib STATIC
           ${RawCVPro_CXX_SOURCES}
           ${RawCVPro_CUDA_SOURCES}
           )
add_executable(RawCvPro OpencvPipeline.cpp)
# add_executable(RawCvPro  ${RawCVPro_CXX_SOURCES} ${RawCVPro_CUDA_SOURCES})
target_link_libraries(RawCvPro PUBLIC ${OpenCV_LIBS} -lcudart -lcudadevrt MyLib)