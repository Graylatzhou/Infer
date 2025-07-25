#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>   
class SiluOperator : public Operator {
public:
    void forward(const Tensor<T>* input, Tensor<T>* output);
    
    OperatorType type() const override { return OperatorType::SILU; }
    std::string name() const override { return "SILU"; }
};
template class infer::SiluOperator<float>;
template class infer::SiluOperator<__nv_bfloat16>;
}

void silu_impl(torch::Tensor& output, const torch::Tensor& input);
void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input);  // [..., 2 * d]