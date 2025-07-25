#pragma once
#include "../operator.hpp"
#include "../tensor.hpp"
#include "../opregistry.hpp"

// namespace infer {
// template <typename T>
// class EmbeddingOperator : public Operator {
// public: 
//     void forward(const Tensor<int32_t>* input, const Tensor<T>* weight, Tensor<T>* output);

//     OperatorType type() const override { return OperatorType::EMBEDDING; }
//     std::string name() const override { return "EMBEDDING"; }
// };
// template class infer::EmbeddingOperator<float>;
// template class infer::EmbeddingOperator<__nv_bfloat16>;
// }
void embedding_impl(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight);
