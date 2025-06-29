#include "operators/rope.hpp"

// q or k [seq_len, num_heads, head_dim]
// sin_table [mas_seq_len, head_dim]
// cos_table [mas_seq_len, head_dim]
template <typename T, typename Tindex>
__global__ void rope_forward_kernel(
    const T* input,
    const Tindex* position_ids,
    T* output,
    const T* sin_table,
    const T* cos_table,
    const int head_dim,
    int seq_stride,
    int num_heads_stride
) {
    int bx = blockIdx.x; // seq_len
    int by = blockIdx.y; // num_heads
    int tx = threadIdx.x;
    int block_offset = bx * seq_stride + by * num_heads_stride;

    int pos_id = position_ids[bx];
    int table_offset = pos_id * head_dim; //决定table的第几行

    int half_dim = head_dim / 2;
    for (int i = tx; i < half_dim; i += blockDim.x) {
        float sin_value = __bfloat162float(sin_table[table_offset + i]);
        float cos_value = __bfloat162float(cos_table[table_offset + i]);
        if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            float x1 = __bfloat162float(input[block_offset + i]);
            float x2 = __bfloat162float(input[block_offset + i + half_dim]);
            output[block_offset + i] = __float2bfloat16(x1 * cos_value - x2 * sin_value);
            output[block_offset + i + half_dim] = __float2bfloat16(x1 * sin_value + x2 * cos_value);
        } else {
            float x1 = input[block_offset + i];
            float x2 = input[block_offset + i + half_dim];
            output[block_offset + i] = x1 * cos_value - x2 * sin_value;
            output[block_offset + i + half_dim] = x2 * cos_value + x1 * sin_value;
        }
    }
}

namespace infer {
template <typename T>
void RopeOperator<T>::forward(std::vector<const Tensor<T>*> input0, Tensor<int32_t>* input1, Tensor<T>* output) {
    const auto input = input0[0];
    const auto position_ids = input1;
    const auto sin_table = input0[1];
    const auto cos_table = input0[2];

    int seq_len = input->shape()[0];
    int num_heads = input->shape()[1];
    int head_dim = input->shape()[2];

    int seq_stride = input->stride()[0];
    int num_heads_stride = input->stride()[1];

    dim3 block(head_dim / 2);
    dim3 grid(seq_len, num_heads);
    
    rope_forward_kernel<T, int32_t><<<grid, block, 0, input->getStream()>>>(
        input->data_ptr(),
        position_ids->data_ptr(),
        output->data_ptr(),
        sin_table->data_ptr(),
        cos_table->data_ptr(),
        head_dim,
        seq_stride,
        num_heads_stride
    );
    
}

}