#pragma once
#include "operators/operator.hpp" 
#include "tensor.hpp"
#include "operators/opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class MulOperator : public Operator<T> {
public:
    void forward(std::vector<const Tensor<T>*> input0, Tensor<T>* output) override;
    
    OperatorType type() const override { return OperatorType::MUL; }
    std::string name() const override { return "MUL"; }
};
REGISTER_OPERATOR(infer::OperatorType::MUL, Mul, infer::MulOperator)
} // namespace infer