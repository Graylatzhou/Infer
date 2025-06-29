#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"

namespace infer {
template <typename T>   
class SoftmaxOperator : public Operator<T> {
public:
    void forward(std::vector<const Tensor<T>*> input0, Tensor<T>* output);
    
    OperatorType type() const override { return OperatorType::SOFTMAX; }
    std::string name() const override { return "SOFTMAX"; }
};

REGISTER_OPERATOR(infer::OperatorType::SOFTMAX, SOFTMAX, infer::SoftmaxOperator)
}