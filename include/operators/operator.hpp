#pragma once
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "tensor.hpp"

namespace infer {

enum class OperatorType {
    MATMUL,
    RELU,
    SOFTMAX,
    SILU,
    FLASHATTENTION,
    ADD_RMS_NORM,

};

// 支持的数据类型
enum class DataType {
    FLOAT32,  // FP32
    FLOAT16,  // FP16
};

// 前置声明模板基类
template <typename T>
class Operator;

using OperatorFP32 = Operator<float>;
using OperatorFP16 = Operator<half>;

template <typename T>
class Operator {
public:
    virtual ~Operator() = default;
    
    virtual OperatorType type() const = 0;

    virtual void forward(std::vector<const Tensor<T>*> input0, Tensor<T>* output) = 0;
    
    virtual std::string name() const = 0;

    DataType dataType() const {
        if constexpr (std::is_same_v<T, float>) {
            return DataType::FLOAT32;
        } else if constexpr (std::is_same_v<T, half>) {
            return DataType::FLOAT16;
        }
        return DataType::FLOAT32;  // 默认值
    }
    
    // 转换为字符串的数据类型名称
    std::string dataTypeName() const {
        switch (dataType()) {
            case DataType::FLOAT32: return "float32";
            case DataType::FLOAT16: return "float16";
            default: return "unknown";
        }
    }
};

// 定义算子的共享指针类型
template <typename T>
using OperatorPtr = std::shared_ptr<Operator<T>>;

} 