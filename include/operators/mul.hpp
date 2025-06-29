#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class MulOperator : public Operator {
public:
    void forward(const Tensor<T>* a, const Tensor<T>* b, Tensor<T>* output);
    
    OperatorType type() const override { return OperatorType::MUL; }
    std::string name() const override { return "MUL"; }

    
private:
};
template class infer::MulOperator<float>;
template class infer::MulOperator<__nv_bfloat16>;
} // namespace infer