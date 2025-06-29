#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class RopeOperator : public Operator {
public:
    void forward(const Tensor<T>* input, Tensor<T>* output, const Tensor<T>* sin_table, 
        const Tensor<T>* cos_table, const Tensor<int32_t>* position_ids);

    OperatorType type() const override { return OperatorType::ROPE; }
    std::string name() const override { return "Rope"; }
};

template class infer::RopeOperator<float>;
template class infer::RopeOperator<__nv_bfloat16>;
} // namespace infer