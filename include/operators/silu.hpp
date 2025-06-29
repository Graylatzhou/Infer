#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>   

class SiluOperator : public Operator<T> {
public:
    void forward(std::vector<const Tensor<T>*> input, Tensor<T>* output);
    
    OperatorType type() const override { return OperatorType::SILU; }
    std::string name() const override { return "SILU"; }
};

REGISTER_OPERATOR(infer::OperatorType::SILU, Silu, infer::SiluOperator)
}