#pragma once
#include "operators/operator.hpp"  // 从include根目录引用
#include "tensor.hpp"
#include "operators/opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class ReluOperator : public Operator<T> {
public:
    void forward(std::vector<const Tensor<T>*> input0, Tensor<T>* output) override;

    OperatorType type() const override { return OperatorType::RELU; }
    std::string name() const override { return "Relu"; }
};

} // namespace infer