#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

// namespace infer {
// template <typename T>
// class RopeOperator : public Operator {
// public:
//     void forward(const Tensor<T>* input, Tensor<T>* output, const Tensor<T>* sin_table, 
//         const Tensor<T>* cos_table, const Tensor<int64_t>* position_ids);

//     OperatorType type() const override { return OperatorType::ROPE; }
//     std::string name() const override { return "Rope"; }
// };

// template class infer::RopeOperator<float>;
// template class infer::RopeOperator<__nv_bfloat16>;
// } // namespace infer

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    std::optional<torch::Tensor> key,
    // null or
    // [batch_size, seq_len, num_kv_heads * head_size] or
    // [num_tokens, num_kv_heads * head_size] or
    // [batch_size, seq_len, num_heads, head_size] or
    // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox);