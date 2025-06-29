cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
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

# cutlass
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
if(NOT EXISTS ${CUTLASS_DIR})
    message(FATAL_ERROR "CUTLASS directory does not exist: ${CUTLASS_DIR}")
endif()
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUDA_INCLUDE_DIRS}
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
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARY}
)

target_include_directories(infer
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CUDA_INCLUDE_DIRS}
    third_party/cutlass/include
    third_party/cutlass/tools/util/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(infer OpenMP::OpenMP_CXX)
endif()

option(BUILD_MAIN "Build main executable" ON)
if(BUILD_MAIN)
  add_executable(infer_app src/main.cu)
  target_link_libraries(infer_app infer)
  target_link_options(infer_app PRIVATE -rdynamic)
endif()