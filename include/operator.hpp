#pragma once
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "tensor.hpp"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace infer {

enum class OperatorType {
    MATMUL,
    RELU,
    SOFTMAX,
    SILU,
    FLASHATTENTION,
    ADD_RMS_NORM,
    MUL,
    ADD,
    ROPE,
    EMBEDDING
};

// 支持的数据类型
enum class DataType {
    FLOAT32,  // FP32
    FLOAT16,  // FP16
};


class Operator {
public:
    virtual ~Operator() = default;

    virtual std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& inputs) = 0;

    virtual OperatorType type() const = 0;
    
    virtual std::string name() const = 0;

};

// 定义算子的共享指针类型

} 

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)
#else
  #define VLLM_LDG(arg) *(arg)
#endif

