cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Infer LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准和架构
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)  # 根据你的目标GPU调整

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")

find_package(OpenMP)

# 找到CUDA和cuBLAS
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "cuBLAS library not found")
endif()

# 输出信息
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "cuBLAS library: ${CUBLAS_LIBRARY}")


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# 显式列出主要操作符目录
set(OPERATOR_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/matmul
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/flashattn
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/softmax
)

# 收集所有源文件
file(GLOB_RECURSE OPERATOR_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*/*.cu"
)

message(STATUS "Operator sources: ${OPERATOR_SOURCES}")

# 创建库
add_library(infer STATIC ${OPERATOR_SOURCES})
target_link_libraries(infer
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARY}
)

# 为库设置包含路径（现代CMake推荐的方式）
target_include_directories(infer
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(infer OpenMP::OpenMP_CXX)
endif()

option(BUILD_MAIN "Build main executable" ON)
if(BUILD_MAIN)
  add_executable(infer_app src/main.cu)
  target_link_libraries(infer_app infer)
endif()