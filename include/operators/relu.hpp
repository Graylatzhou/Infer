#pragma once
#include "operator.hpp"  // 从include根目录引用
#include "tensor.hpp"
#include "opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class ReluOperator : public Operator {
public:
    void forward(const Tensor<T>* input, Tensor<T>* output);

    OperatorType type() const override { return OperatorType::RELU; }
    std::string name() const override { return "Relu"; }
};

} // namespace infer