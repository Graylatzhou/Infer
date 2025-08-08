#include <torch/extension.h>
#include "Inferop.hpp"

TORCH_LIBRARY(_C, m) {  // ✅ 直接使用 _C 而不是 TORCH_EXTENSION_NAME
    m.def("add(Tensor a, Tensor b, Tensor! output) -> ()");
    m.def("rms_norm(Tensor input, Tensor weight, Tensor! output, Tensor? bias, float eps) -> ()");
    m.def("matmul(Tensor input, Tensor weight, Tensor! output) -> ()");
    m.def("embedding(Tensor! output, Tensor input, Tensor weight) -> ()");
    m.def("flash_attn_prefill(Tensor Q, Tensor K, Tensor V, Tensor! O) -> ()");
    m.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
    m.def("silu(Tensor! output, Tensor input) -> ()");
    m.def("softmax(Tensor! output, Tensor input, int axis) -> ()");
    m.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
    m.def(
      "rms_norm_vllm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");
    m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
}

TORCH_LIBRARY_IMPL(_C, CUDA, m) { 
    m.impl("add", torch::kCUDA, &add_impl);
    m.impl("rms_norm", torch::kCUDA, &rms_norm_impl);
    m.impl("matmul", torch::kCUDA, &matmul_impl);
    m.impl("embedding", torch::kCUDA, &embedding_impl);
    m.impl("flash_attn_prefill", torch::kCUDA, &flash_attn_prefill_impl);
    m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);
    m.impl("silu", torch::kCUDA, &silu_impl);
    m.impl("softmax", torch::kCUDA, &softmax_impl);
    m.impl("rms_norm_vllm", torch::kCUDA, &rms_norm);
    m.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);
    m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
}
