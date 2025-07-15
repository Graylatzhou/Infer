#include "operators/flashattn.hpp"

// grid (batch_size, nh)
template <typename T, int Bc, int Br>
__global__ void flash_attn_cuda_kernelv1(const T* Q, const T* K, const T* V, T* output, 
                                    int batch_size, int nh, int seq_len, int head_dim,
                                    int Tc, int Tr, float softmax_scale) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    const int tile_size_KV = Bc * head_dim;
    const int tile_size_Q = Br * head_dim;
    const int tile_size_lm = batch_size * nh * seq_len;
    extern __shared__ T smem[];
    T* smem_Q = smem;
    T* smem_K = &smem[tile_size_Q];
    T* smem_V = &smem_K[tile_size_KV];
    T* smem_O = &smem_V[Bc * Br];
    float* smem_l = reinterpret_cast<float*>(smem_V + tile_size_KV);
    float* smem_m = reinterpret_cast<float*>(smem_l + tile_size_lm);
    const int qkv_offset = (bx + by * gridDim.x) * seq_len * head_dim;
    const int lm_offset = (bx + by * gridDim.x) * seq_len;

    // 总的分块逻辑描述：
    // 首先是QKV都是把batch_size * nh 作为外层，这两个维度不参与计算
    // 一个块 BC个线程，每个线程load head_dim个元素
    // outter loop over TC
    for (int i = 0; i < Tc; i++) {
        for (int j = 0; j < head_dim; j++) {
            // i * Bc决定是哪个块
            smem_K[tid * head_dim + j] = K[qkv_offset + (i * Bc + tid) * head_dim + j];
            smem_V[tid * head_dim + j] = V[qkv_offset + (i * Bc + tid) * head_dim + j];
        }
        for (int j = 0; j < Tr; j++) {
            // inner loop over Tr 也就是一个KV块和所有的Q计算得到一块output
            for (int k = 0; k < head_dim; k++) {
                smem_Q[tid * head_dim + k] = Q[qkv_offset + (j * Br + tid) * head_dim + k];
            }
            __syncthreads();
            // max也就是一个head_dim中的最大值，然后我需要定位偏移
            // 一个block对应Bc分块，所以一个线程也就对应head_dim个元素的操作量，也就对应l，m中的一个元素
            float prev_m_val = smem_m[lm_offset + i * Bc + tid];
            float prev_sum_val = smem_l[lm_offset + i * Bc + tid];
            float row_current_block_max = -INFINITY;
            // 0 1head_dim 2head_dim 3head_dim 4head_dim
            // 计算Q和K的点积
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < Br; x++) {
                    // tid从0到blockDim.x在列方向上遍历
                    // Q[tid][x] * K[y][x]
                    sum += __bfloat162float(smem_Q[(tid * head_dim) + x]) * __bfloat162float(smem_K[(y * head_dim) + x]);
                }
                sum *= softmax_scale;
                smem_O[(Bc * tid) + y] = __float2bfloat16(sum);
                if (sum > row_current_block_max) row_current_block_max = sum;
            }

            float row_current_block_l = 0;
            for (int y = 0; y < Bc; y++) {
                float tmp = __expf(__bfloat162float(smem_O[(Bc * tid) + y]) - row_current_block_max);
                smem_O[(Bc * tid) + y] = __float2bfloat16(tmp);
                row_current_block_l += tmp;
            }

            float row_new_m = max(prev_m_val, row_current_block_max);
            float row_new_l = (__expf(prev_m_val - row_new_m) * prev_sum_val) + 
                (__expf(row_current_block_max - row_new_m) * row_current_block_l);

            for (int x = 0; x < head_dim; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++) {
                    pv += __bfloat162float(smem_O[(Bc * tid) + y]) * __bfloat162float(smem_V[(y * head_dim) + x]);
                }
                float result = __bfloat162float((1 / row_new_l) * (prev_sum_val * __expf(prev_m_val - row_new_m) 
                    * output[qkv_offset + i * Bc * Br + tid * head_dim +x]) + (__expf(row_current_block_max - row_new_m) * pv));
                output[qkv_offset + i * Bc * Br + tid * head_dim + x] = result;
                
            }
        }
        __syncthreads();
    }
}


namespace infer {
// 单batch
// Q_weight
template <typename T>
void FlashAttnOperator<T>::forward(const Tensor<T>* Q, const Tensor<T>* K, const Tensor<T>* V, Tensor<T>* output) {
    const int batch_size = Q->shape()[0];
    const int nh = Q->shape()[1];
    const int seq_len = Q->shape()[2];
    const int head_dim = Q->shape()[3];

    constexpr int Bc = 32;
    constexpr int Br = 32;
    const int Tc = (seq_len + Bc - 1) / Bc; // kv
    const int Tr = (seq_len + Br - 1) / Br;
    // temp l,
    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    // Qi, Kj, Vj, l, m
    const int smem_size = (Bc * head_dim * 2 + Br * head_dim + Bc * Br) * sizeof(T); + batch_size * nh * seq_len * 2 * sizeof(float);
    dim3 grid(batch_size, nh);
    dim3 block(Bc);
}
}