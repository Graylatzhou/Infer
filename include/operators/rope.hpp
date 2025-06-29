#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class RopeOperator : public Operator<T> {
public:
    void forward(std::vector<const Tensor<T>*> input0, Tensor<int32_t>* input2, Tensor<T>* output);

    OperatorType type() const override { return OperatorType::ROPE; }
    std::string name() const override { return "Rope"; }
};
REGISTER_OPERATOR(infer::OperatorType::ROPE, ROPE, infer::RopeOperator);

} // namespace infer