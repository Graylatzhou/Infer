#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class MulOperator : public Operator<T> {
public:
    void forward(std::vector<const Tensor<T>*> input0, Tensor<T>* output);
    
    OperatorType type() const override { return OperatorType::MUL; }
    std::string name() const override { return "MUL"; }

    
private:

};
REGISTER_OPERATOR(infer::OperatorType::MUL, Mul, infer::MulOperator)
} // namespace infer