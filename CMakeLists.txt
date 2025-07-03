cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 在 project 语句之前，设置 CUDA 编译器路径（确保指向 12.8 版本）
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

project(Infer LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
find_package(OpenMP)

# 防止优化掉全局初始化变量
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
endif()

# 使用现代 CMake 方法查找 CUDA Toolkit
find_package(CUDAToolkit 12.0 REQUIRED)

# 输出信息以确认正确的版本
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit root directory: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARIES}")

# cutlass
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
if(NOT EXISTS ${CUTLASS_DIR})
    message(FATAL_ERROR "CUTLASS directory does not exist: ${CUTLASS_DIR}")
endif()
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 显式列出主要操作符目录
set(OPERATOR_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/matmul
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/flashattn
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/softmax
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/add
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/silu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/mul    
)

# 收集所有源文件
file(GLOB_RECURSE OPERATOR_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*/*.cu"
)

message(STATUS "Operator sources: ${OPERATOR_SOURCES}")
# 打印每个文件以确认
foreach(SOURCE_FILE ${OPERATOR_SOURCES})
  message(STATUS "  Source: ${SOURCE_FILE}")
endforeach()

# 创建库
add_library(infer STATIC ${OPERATOR_SOURCES})
set_target_properties(infer PROPERTIES 
  ENABLE_EXPORTS ON
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(infer
    CUDA::cudart
    CUDA::cublas
    CUDA::cuda_driver
)

target_include_directories(infer
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    third_party/cutlass/include
    third_party/cutlass/tools/util/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(infer OpenMP::OpenMP_CXX)
endif()

option(BUILD_MAIN "Build main executable" ON)
if(BUILD_MAIN)
  add_executable(infer_app src/main.cu)
  target_link_libraries(infer_app infer CUDA::cuda_driver)
  target_link_options(infer_app PRIVATE -rdynamic)
endif()