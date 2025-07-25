#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

// namespace infer {
// template <typename T>
// class AddOperator : public Operator {
// public:
//     void forward(const Tensor<T>* a, const Tensor<T>* b, Tensor<T>* output);
    
//     OperatorType type() const override { return OperatorType::ADD; }
//     std::string name() const override { return "ADD"; }
// };
// template class infer::AddOperator<float>;
// template class infer::AddOperator<__nv_bfloat16>;


// }
void add_impl(const torch::Tensor& a, const torch::Tensor& b, torch::Tensor& output); 
