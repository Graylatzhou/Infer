#pragma once
#include "../operations.hpp"
#include "../tensor.hpp"
#include "../opregistry.hpp"

namespace infer {
template <typename T>
class FlashAttnOperator : public Operator {
public:
    void forward(const Tensor<T>* Q, const Tensor<T>* K, const Tensor<T>* V,
                 Tensor<T>* output);

    OperatorType type() const override { return OperatorType::FLASHATTENTION; }
    std::string name() const override { return "FLASHATTENTION"; }
}
template class infer::FlashAttnOperator<float>;
template class infer::FlashAttnOperator<__nv_bfloat16>;
}