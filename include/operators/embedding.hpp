#pragma once
#include "../operations.hpp"
#include "../tensor.hpp"
#include "../opregistry.hpp"

namespace infer {
template <typename T>
class EmbeddingOperator : public Operator {
public: 
    void forward(const Tensor<T>* input, const Tensor<T>* weight, Tensor<T>* output);

    OperatorType type() const override { return OperatorType::EMBEDDING; }
    std::string name() const override { return "EMBEDDING"; }
};
template class infer::EmbeddingOperator<float>;
template class infer::EmbeddingOperator<__nv_bfloat16>;
}