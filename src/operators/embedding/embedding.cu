#include "operators/embedding.hpp"

#define FLOAT4_OUTPUT(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT4_INPUT(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define BFLOAT4_OUTPUT(value) (reinterpret_cast<float2 *>(&(value))[0])
#define BFLOAT4_INPUT(value) (reinterpret_cast<const float2 *>(&(value))[0])

template <typename T>
__global__ void embedding_forward_kernel(const int64_t* input, const T* weight, T* output, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int offset = input[bx] * emb_size;
    if constexpr (std::is_same<T, float>::value) {
        FLOAT4_OUTPUT(output[bx * emb_size + 4 * tx]) = FLOAT4_INPUT(weight[offset + 4 * tx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        BFLOAT4_OUTPUT(output[bx * emb_size + 4 * tx]) = BFLOAT4_INPUT(weight[offset + 4 * tx]);
    }
}

void embedding_impl(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight) {
    int seq_len = input.numel();
    int emb_size = weight.size(1);
    dim3 block(emb_size / 4, 1, 1);
    dim3 grid(seq_len, 1, 1);
    c10::cuda::OptionalCUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (weight.dtype() == torch::kBFloat16) {
        embedding_forward_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<int64_t*>(input.data_ptr()), reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()), reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), emb_size);
    } else if (weight.dtype() == torch::kFloat32) {
        embedding_forward_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<int64_t*>(input.data_ptr()), reinterpret_cast<float*>(weight.data_ptr()), reinterpret_cast<float*>(output.data_ptr()), emb_size);
    } else {
        throw std::runtime_error("Unsupported data type for embedding");
    }
}

// namespace infer {
// template <typename T>
// void EmbeddingOperator<T>::forward(const Tensor<int32_t>* input, const Tensor<T>* weight,
//                                    Tensor<T>* output) {
//     auto seq_len = input->shape()[0];
//     auto emb_size = weight->shape()[1];
//     // dim3 block(emb_size / 4, 1, 1);
//     // dim3 grid(seq_len, 1, 1);
//     // embedding_forward_kernel<T><<<grid, block, 0, input->getStream()>>>(
//     //     input->data_ptr(), weight->data_ptr(), output->data_ptr(), emb_size);
// }
// }