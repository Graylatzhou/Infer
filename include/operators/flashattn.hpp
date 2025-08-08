#pragma once
#include "../operator.hpp"
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
};
template class infer::FlashAttnOperator<float>;
template class infer::FlashAttnOperator<__nv_bfloat16>;
}

void flash_attn_prefill_impl(const torch::Tensor& Q, const torch::Tensor& K, 
                             const torch::Tensor& V, torch::Tensor& O);

std::vector<torch::Tensor> flash_attention_v2_cutlass(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, bool is_causal = false, double softmax_scale=1);
