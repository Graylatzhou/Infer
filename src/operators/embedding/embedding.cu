#include "embedding.hpp"

#define FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define BFLOAT4(value) (reinterpret_cast<const float2 *>(&(value))[0])

template <typename T>
__global__ void embedding_forward_kernel(const T* input, const T* weight, T* output, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int offset = input[bx] * emb_size;
    if constexpr (std::is_same<T, float>::value) {
        FLOAT4(output[bx * emb_size + 4 * tx]) = FLOAT4(weight[offset + 4 * tx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        BFLOAT4(output[bx * emb_size + 4 * tx]) = BFLOAT4(weight[offset + 4 * tx]);
    }
}

namespace infer {
template <typename T>
void EmbeddingOperator<T>::forward(const Tensor<T>* input, const Tensor<T>* weight,
                                   Tensor<T>* output) {
    auto seq_len = input->shape()[0];
    auto emb_size = weight->shape()[1];
    dim3 block(emb_size / 4, 1, 1);
    dim3 grid(seq_len, 1, 1);
    embedding_forward_kernel<T><<<grid, block, 0, input->stream()>>>(
        input->data_ptr(), weight->data_ptr(), output->data_ptr(), emb_size);
}
}