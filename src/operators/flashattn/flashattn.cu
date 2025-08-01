#include "operators/flashattn.hpp"

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                                \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
        "l"(src), "n"(bytes))

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])

#define LDST128BITSCONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define LDST32BITSCONST(value) (reinterpret_cast<const half2 *>(&(value))[0])

#define DEVICE_INLINE __device__ __inline__
#define WARP_SIZE 32
template <typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_sum(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, kWarpSize);
  }
  return val;
}

template <typename T, const int kWarpSize = WARP_SIZE>
DEVICE_INLINE T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
  }
  return val;
}

template <typename T, int Br, int Bc, int kStage, int bit_offset, int Pads> // for headdim=128
__global__ void flash_attention_v2_kernel(const T* Q, const T* K, const T* V, T* output,
                                       int nh_q, int nh_kv, int seq_len, float softmax_scale,
                                       int q_offset_causal, int kv_cache_lenth) {
  float thr_row_max = -INFINITY;
  float thr_row_sum = 0.0f;
  float acc_O[Br] = {0.0f};
  float acc_D[Br] = {0.0f};

  int QKV_batch_id = blockIdx.y / nh_q;
  int QKV_head_id = blockIdx.y % nh_q;
  int Q_tile_id = blockIdx.x;
  int tid = threadIdx.x;
  const int Tc = (seq_len + Bc - 1) / Bc;
  constexpr int Q_tile_size = Br * (128 + Pads);
  constexpr int K_tile_size = Bc * (128 + Pads);
  extern __shared__ T smem[];
  T* Q_tile_smem = smem; 
  T* K_tile_smem = Q_tile_smem + Q_tile_size;
  T* V_tile_smem = K_tile_smem + K_tile_size * kStage;
  float* S_tile_smem = reinterpret_cast<float*>(V_tile_smem + K_tile_size);
  
  // thread per row = threadnum / Br
  // 32 * 128
  // #define THREADS_PER_ROW (128 / Br)
  int load_smem_Q_Br = tid / (128 / Br);
  int load_smem_Q_hd = (tid % (128 / Br)) << bit_offset; // << 4 = *16

  int load_smem_K_Bc = tid / (128 / Bc);
  int load_smem_K_hd = (tid % (128 / Bc)) << bit_offset; // << 4 = *16

  int load_smem_V_Bc = tid / (128 / Bc);
  int load_smem_V_hd = (tid % (128 / Bc)) << bit_offset; // << 4 = *16
  // Q br dimension = tile_id * tile_size + partial location
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= seq_len) return;
  int kv_head_id = QKV_head_id / (nh_q / nh_kv);
  int load_gmem_Q_offset = (QKV_batch_id * nh_q + QKV_head_id) * seq_len * 128;
  int load_gmem_KV_offset = (QKV_batch_id * nh_kv + kv_head_id) * seq_len * 128;

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // 因为一个内核代码可以看作是一个block中的行为描述
  // 这个block只需要加载一次对应的Q分块，而需要加载多次KV分块，因为还需要在KV分块的seq_len维度上遍历
  {
    int load_gmem_Q_hd = load_smem_Q_hd;
    int load_gmem_Q_addr = load_gmem_Q_offset + load_gmem_Q_Br * 128 + load_gmem_Q_hd;
    uint32_t load_smem_Q_ptr = smem_Q_base_ptr + (load_smem_Q_Br * (128 + Pads) + load_smem_Q_hd) * sizeof(T); // bytes
    #pragma unroll
    for (int i = 0; i < 128 / (128 / Br); i += 8) {
      CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }
  // 以上完成了Q分块异步加载的任务提交
  if constexpr (kStage > 1) {
#pragma unroll
    for (int i = 0; i < kStage; i++) {
      int load_gmem_K_hd = load_smem_K_hd;
      int load_gmem_K_Bc = i * Bc + load_smem_K_Bc; // total BC
      int load_gmem_K_addr = load_gmem_KV_offset + load_gmem_K_Bc * 128 + load_gmem_K_hd;
      // (128 + Pads) = (head_dim + Pads)
      uint32_t load_smem_K_ptr = smem_K_base_ptr + (i * K_tile_size + load_smem_K_Bc * (128 + Pads) + load_smem_K_hd) * sizeof(T); // bytes
      #pragma unroll
      for (int j = 0; j < 128 / (128 / Bc); j += 8)
      {
        CP_ASYNC_CG(load_smem_K_ptr + j * 2, &K[load_gmem_K_addr + j], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }
  } // 以上完成了prefetch的K分块
  CP_ASYNC_WAIT_GROUP(kStage - 2); // s2->0, s3->1, s4->2
  __syncthreads();

  #pragma unroll 1
  for (int tile_K_id = 0; tile_K_id < Tc; tile_K_id++) {
    int smem_sel = tile_K_id % kStage;
    int smem_sel_next = (tile_K_id + (kStage - 1)) % kStage;
    // 在执行后续计算前先prefetch V
    // 这样在后续计算的同时也会加载V分块
    if constexpr (kStage > 1) {
      // prefetch tile V
      {
        int load_gmem_V_hd = load_smem_V_hd;
        int load_gmem_V_Bc = load_smem_K_Bc + tile_K_id * Bc; // total BC
        int load_gmem_V_addr = load_gmem_KV_offset + load_gmem_V_Bc * 128 + load_gmem_V_hd;
        uint32_t load_smem_V_ptr = smem_V_base_ptr + (load_smem_V_Bc * (128 + Pads) + load_smem_V_hd) * sizeof(T); // bytes
        #pragma unroll
        for (int i = 0; i < 128 / (128 / Bc); i += 8)
        {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }

      if ((tile_K_id + 1) < Tc) {
        // 预取下次的K分块如果还不是最后一个分块
        int load_gmem_K_hd = load_smem_K_hd;
        int load_gmem_K_Bc = (tile_K_id + 1) * Bc + load_smem_K_Bc;
        int load_gmem_K_addr = load_gmem_KV_offset + load_gmem_K_Bc * 128 + load_gmem_K_hd;
        uint32_t load_smem_K_ptr = smem_K_base_ptr + (smem_sel_next * K_tile_size + load_smem_K_Bc * (128 + Pads) + load_smem_K_hd) * sizeof(T); // bytes
        #pragma unroll
        for (int i = 0; i < 128 / (128 / Bc); i += 8)
        {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    } 
    // load到寄存器做计算 或者直接根据线程划分任务在smem中计算
    // tile_Q [16, 128] tile_K [16, 128]
    // -> tile_Q @ tile_K^T = [16, 16]
    // 总共 Br * Bc个输出，Br * Bc -> 一个线程负责Br * Bc / 128个输出
    // tid % (128 / Br)得到当前线程在一行上的坐标
    // 需要左移之后占满 Bc个位置
    // 128 / Br得到一行的线程数目，Bc / thr_num得到每个线程负责的列数
    int pos_thr_calc_x = (tid % (128 / Br)) * (Bc / (128 / Br)); // 
    // 0, 2, 4, 6, 8, 10, 12, 14
    int pos_thr_calc_y = tid / (128 / Br); // 0-15
    // 这里计算出了psum在Score中的位置
    // 如果在Score中的位置是[1, 1],即Q_tile_smem[]
    float psum[Bc / (128 / Br)] = {0.0f};
    for (int j = 0; j < 128; j++) {
      #pragma unroll
      for (int i = 0; i < Bc / (128 / Br); i++) {
        psum[i] += __bfloat162float(Q_tile_smem[pos_thr_calc_y * (128 + Pads) + j]) * __bfloat162float(K_tile_smem[smem_sel * K_tile_size + (pos_thr_calc_x + i) * (128 + Pads) + j]);
      }
    }
    #pragma unroll
    for (int i = 0; i < Bc / (128 / Br); i++) {
      psum[i] *= softmax_scale;
    }
    #pragma unroll
    for (int i = 0; i < Bc / (128 / Br); i++) {
      psum[i] = tile_K_id * Bc + pos_thr_calc_x + i > load_gmem_Q_Br ? -INFINITY : psum[i];
    }
    float local_max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < Bc / (128 / Br); i++) {
      local_max = max(local_max, psum[i]);
    }
    float current_max = warp_reduce_max<float, 4>(local_max);
    float new_max = max(current_max, thr_row_max);

    #pragma unroll
    for (int i = 0; i < Bc / (128 / Br); i++) {
      psum[i] = __expf(psum[i] - new_max);
    }

    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < Bc / (128 / Br); i++) {
      local_sum += psum[i];
    }
    float current_sum = warp_reduce_sum<float, 4>(local_sum);
    // 添加mask
    // 避免Q看见未来的K
    // Q[seq_len, head_dim]
    for (int i = 0; i < Bc / (128 / Br); i++) {
      // 我只需要知道token offset即可
      // int q_offset = load_gmem_Q_Br 即为token offset
      // 根据tile_K_id可以知道是在哪个K分块
      // int k_offset = tile_K_id * Bc + pos_thr_calc_x + i;
      S_tile_smem[pos_thr_calc_y * Bc + pos_thr_calc_x + i] = psum[i];
    }

    if constexpr (kStage > 1) {
      if (tile_K_id + 1 < Tc) {
        CP_ASYNC_WAIT_GROUP(1);
      } else {
        CP_ASYNC_WAIT_GROUP(0);
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();
    // S[Br, Bc] = [16, 16]
    // V[Bc, 128] = [16, 128]
    // O[Br, 128] = [16, 128] 128个线程，每个线程持有Br个值
    // O[thr_O_row, thr_O_col] = S[thr_O_row, i:i+15] @ V[i:i+15, thr_O_col]
    int thr_O_row = load_smem_Q_Br; // tid / (128 / Br);
    int thr_O_col = load_smem_Q_hd; // (tid % (128 / Br)) << 4; // << 4 = *16
    #pragma unroll
    for (int i = 0; i < Br; i++) {
      acc_D[i] = 0.0f;
    }
    #pragma unroll
    for (int j = 0; j < Br; j++) {
      #pragma unroll
      for (int k = 0; k < Bc; k++) {
        acc_D[j] += __bfloat162float(V_tile_smem[k * (128 + Pads) + thr_O_col + j]) * S_tile_smem[thr_O_row * Bc + k];
      }
    }
    // 计算出O后需要更新根据max和sum 以及修正o
    float rescale_factor = __expf(thr_row_max - new_max);
    #pragma unroll
    for (int j = 0; j < Br; j++) {
      acc_O[j] = __fmaf_rn(rescale_factor, acc_O[j], acc_D[j]); // O(i) = O(i-1) * exp(m(i-1) - m(i)) + D(i)
    }
    // sum和max是每8个线程持有一个相同的值
    // rescale sum
    thr_row_sum = __fmaf_rn(rescale_factor, thr_row_sum, current_sum);
    thr_row_max = new_max;
    // 确保下个循环的k tile准备好了
    if constexpr (kStage > 1) {
      if (tile_K_id + 1 < Tc) {
        CP_ASYNC_WAIT_GROUP(0);
      } 
      __syncthreads();
    }
  } 
  // 最后rescale 输出
  float rescale_factor = 1.0f / thr_row_sum;
  T R_out[Br];
  #pragma unroll
  for (int i = 0; i < Br; i++) {
    R_out[i] = __float2bfloat16(acc_O[i] * rescale_factor); // O(i) = O(i) / sum
  }
  // gmemoffset确定batch和head的偏移, Q_tile_id确定在哪个分块
  int store_gmem_O_addr = load_gmem_Q_offset + Q_tile_id * Br * 128 + load_smem_Q_Br * 128 + load_smem_Q_hd;
  #pragma unroll
  for (int i = 0; i < Br; i+=8) {
    LDST128BITS(output[store_gmem_O_addr + i]) = LDST128BITS(R_out[i]);
  }
}

__host__ __inline__ int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
void flash_attn_prefill_impl(const torch::Tensor& Q, const torch::Tensor& K, 
                             const torch::Tensor& V, torch::Tensor& O) {
  // 确保输入输出tensor在同一设备上
  TORCH_CHECK(Q.device().is_cuda() && K.device().is_cuda() && V.device().is_cuda() && O.device().is_cuda(),
              "All tensors must be on the same CUDA device.");

  int Batch = Q.size(0);
  int seq_len = Q.size(1);
  int num_heads = Q.size(2);
  int head_dim = Q.size(3);

  int num_heads_kv = K.size(2);

  c10::cuda::OptionalCUDAGuard device_guard(Q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  const int Pads = 8;                // padding size
  const int Bc = 32;                 // K/V 分块大小
  const int Br = 32;                 // Q 分块大小
  const int Tc = (seq_len + Bc - 1) / Bc;  // K/V 分块数量
  const int Tr = (seq_len + Br - 1) / Br;  // Q 分块数量
  constexpr int Q_tile_size = Br * (128 + Pads); 
  constexpr int K_tile_size = Bc * (128 + Pads);

  const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  // (seq_len, batch * num_heads)
  dim3 grid(div_ceil(seq_len, Br), Batch * num_heads);
  dim3 block(128); // 4/8 warps per block
  // 启动kernel
  if (seq_len > 256) {
    const int smem_size = (Q_tile_size + K_tile_size * 3 + K_tile_size) * sizeof(__nv_bfloat16) + Br * Bc * sizeof(float);
    cudaFuncSetAttribute(flash_attention_v2_kernel<__nv_bfloat16, Br, Bc, 3, 5, 8>, 
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    flash_attention_v2_kernel<__nv_bfloat16, Br, Bc, 3, 5, 8><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        num_heads, num_heads_kv, seq_len, softmax_scale, 0, 0
    );
  } else {
    const int smem_size = (Q_tile_size + K_tile_size * 2 + K_tile_size) * sizeof(__nv_bfloat16) + Br * Bc * sizeof(float);
    cudaFuncSetAttribute(flash_attention_v2_kernel<__nv_bfloat16, Br, Bc, 2, 5, 8>, 
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    flash_attention_v2_kernel<__nv_bfloat16, Br, Bc, 2, 5, 8><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        num_heads, num_heads_kv, seq_len, softmax_scale, 0, 0
    );
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in flash_attn_prefill_impl: " + std::string(cudaGetErrorString(err)));
  }
  

}


// namespace infer {
// // 单batch
// // Q_weight
// template <typename T>
// void FlashAttnOperator<T>::forward(const Tensor<T>* Q, const Tensor<T>* K, const Tensor<T>* V, Tensor<T>* output) {
//     const int B = Q->shape()[0];
//     const int nh = Q->shape()[1];
//     const int seq_len = Q->shape()[2];
//     const int head_dim = Q->shape()[3];

//     const int Pads = 8;                // padding size
//     const int Bc = 32;                 // K/V 分块大小
//     const int Br = 32;                 // Q 分块大小
//     const int Tc = (seq_len + Bc - 1) / Bc;  // K/V 分块数量
//     const int Tr = (seq_len + Br - 1) / Br;  // Q 分块数量
//     constexpr int Q_tile_size = Br * (128 + Pads); 
//     constexpr int K_tile_size = Bc * (128 + Pads);

//     const float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
//     const int smem_size = (Q_tile_size + K_tile_size * 2 + K_tile_size) * sizeof(__nv_bfloat16) + Br * Bc * sizeof(float);

//     // (seq_len, batch * num_heads)
//     dim3 grid(div_ceil(seq_len, Br), B * nh);
//     dim3 block(128); // 4/8 warps per block
//     // 启动kernel
//     cudaFuncSetAttribute(flash_attention_v2_kernel<__nv_bfloat16, Br, Bc, 2, 5, 8>, 
//                                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
//     // flash_attention_v2_kernel<__nv_bfloat16, Br, Bc, 2, 5, 8><<<grid, block, smem_size>>>(
//     //     reinterpret_cast<__nv_bfloat16*>(Q.data_ptr()),
//     //     reinterpret_cast<__nv_bfloat16*>(K.data_ptr()),
//     //     reinterpret_cast<__nv_bfloat16*>(V.data_ptr()),
//     //     reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
//     //     nh, nh, seq_len, softmax_scale, 0, 0
//     // );
// }
// }