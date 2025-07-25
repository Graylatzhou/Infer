#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"

// namespace infer {
// template <typename T>   
// class SoftmaxOperator : public Operator {
// public:
//     void forward(Tensor<T>* input0, Tensor<T>* output, int32_t axis);
    
//     OperatorType type() const override { return OperatorType::SOFTMAX; }
//     std::string name() const override { return "SOFTMAX"; }
// };

// template class infer::SoftmaxOperator<float>;
// template class infer::SoftmaxOperator<__nv_bfloat16>;
// }

void softmax_impl(torch::Tensor& output, const torch::Tensor& input, int64_t axis);