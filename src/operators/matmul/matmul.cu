#include "operators/matmul.hpp"

namespace infer {

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
    asm volatile(                                                                \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                                \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
        "l"(src), "n"(bytes))

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])

#define LDST128BITSCONST(value) (reinterpret_cast<const float4 *>(&(value))[0])
#define LDST32BITSCONST(value) (reinterpret_cast<const half2 *>(&(value))[0])

#define LDMATRIX_X2_T(R0, R1, addr)                                              \
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
                : "=r"(R0), "=r"(R1)                                            \
                : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
    asm volatile(                                                                \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
        : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
    asm volatile(                                                                \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
        : "=r"(RD0), "=r"(RD1)                                                   \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))


template <int BM=128, int BN=128, int BK=16, bool BLOCK_SWIZZLE=false>
__global__ void gemm_mma_vectorized_kernel(const half *A, const half *B, half *C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    __shared__ half smemA[BM][BK];
    __shared__ half smemB[BK][BN];
    __shared__ half smemC[BM][BN];
    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id % 2;
    int warp_n = warp_id / 2;
    int lane_id = threadIdx.x % 32;
    // load tileA smem idx calculation
    // 128 * 16 layout
    // 16 elements per row
    int load_smem_a_row = threadIdx.x / 2;
    int load_smem_a_col = (threadIdx.x & 1) << 3;
    // load tileB smem idx calculation
    // 16 * 128 layout
    // 128 / 8 = 16 threads per row
    int load_smem_b_row = threadIdx.x / 16;
    int load_smem_b_col = (threadIdx.x & 15) << 3; // 0 - 15 * 8
    int load_gmem_a_row = by * BM + load_smem_a_row;
    int load_gmem_b_col = bx * BN + load_smem_b_col;

    if (load_gmem_a_row >= M || load_gmem_b_col >= N) return;

    uint32_t RC[4][4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        RC[i][j][0] = 0;
        RC[i][j][1] = 0;
    }
    }
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    // load tileA
    int load_gmem_a_col = bk * BK + load_smem_a_col;
    int load_gmem_b_row = bk * BK + load_smem_b_row;
    int load_gmem_a_addr = load_gmem_a_row * K + load_gmem_a_col;
    int load_gmem_b_addr = load_gmem_b_row * N + load_gmem_b_col;
    LDST128BITS(smemA[load_smem_a_row][load_smem_a_col]) = LDST128BITSCONST(A[load_gmem_a_addr]);
    LDST128BITS(smemB[load_smem_b_row][load_smem_b_col]) = LDST128BITSCONST(B[load_gmem_b_addr]);
    __syncthreads();
    uint32_t RA[4][4];
    uint32_t RB[4][2];
    // ldmatrixA
    // m 方向2个warp 负责 128行加载 -> 一个warp64行 64 / 4
    // WARP_M = 4 即为RC在M方向的重复次数
    #pragma unroll  
    for (int i = 0; i < 4; i++) { // warp_m * MMA_M * WARP_M
        int lane_load_smem_a_row = warp_m * 16 * 4 + i * 16 + lane_id % 16;
        int lane_load_smem_a_col = (lane_id / 16) * 8;// 0-15 -> 0, 16-31 -> 8
        if (lane_load_smem_a_row >= 128) printf("error");
        uint32_t smemA_addr = __cvta_generic_to_shared(&smemA[lane_load_smem_a_row][lane_load_smem_a_col]);
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smemA_addr);
    }
    // n 方向 4个warp，128列加载 -> 一个warp 32列加载，其中一个warp一次加载8列
    // 所以一个warp要加载4次
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int lane_load_smem_b_row = lane_id % 16; //每个线程提供一行smem的首地址 
        int lane_load_smem_b_col = warp_n * 32 + i * 8; 
        uint32_t smemB_addr = __cvta_generic_to_shared(&smemB[lane_load_smem_b_row][lane_load_smem_b_col]);
        LDMATRIX_X2_T(RB[i][0], RB[i][1], smemB_addr);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
        HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
        }
    }
    __syncthreads();
    }
    // store tileC
    // tileC 128 * 128
    #pragma unroll
    for (int i = 0; i < 4; i++) {

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        // lane_id / 4 -> lane_id 0-32 -> 0-8
        int store_smem_c_row = warp_m * 64 + i * 16 + lane_id / 4; //分为两部分存储，因为RC[2]中的两个元素相隔8行
        int store_smem_c_col = warp_n * 32 + j * 8 + (lane_id % 4) * 2;  // 32bits
        LDST32BITS(smemC[store_smem_c_row + 0][store_smem_c_col]) = LDST32BITS(RC[i][j][0]);
        LDST32BITS(smemC[store_smem_c_row + 8][store_smem_c_col]) = LDST32BITS(RC[i][j][1]);
    }
    }
    for (int tid = threadIdx.x; tid < BM * BN; tid += blockDim.x) {
    int store_smem_c_row = tid / BN;
    int store_smem_c_col = tid % BN;
    int store_gmem_c_row = by * BM + store_smem_c_row;
    int store_gmem_c_col = bx * BN + store_smem_c_col;
    if (store_gmem_c_row < M && store_gmem_c_col < N) {
        C[store_gmem_c_row * N + store_gmem_c_col] += smemC[store_smem_c_row][store_smem_c_col];
    }
    }
}      


template <int BM=128, int BN=128, int BK=16, bool BLOCK_SWIZZLE=false, int KStage=2>
__global__ void gemm_mma_async_vectorized_kernel(const half *A, const half *B, half *C, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  __shared__ half smemA[KStage][BM][BK];
  __shared__ half smemB[KStage][BK][BN];
  __shared__ half smemC[BM][BN];
  int warp_id = threadIdx.x / 32;
  int warp_m = warp_id % 2;
  int warp_n = warp_id / 2;
  int lane_id = threadIdx.x % 32;
  // load tileA smem idx calculation
  // 128 * 16 layout
  // 16 elements per row
  int load_smem_a_row = threadIdx.x / 2;
  int load_smem_a_col = (threadIdx.x & 1) << 3;
  // load tileB smem idx calculation
  // 16 * 128 layout
  // 128 / 8 = 16 threads per row
  int load_smem_b_row = threadIdx.x / 16;
  int load_smem_b_col = (threadIdx.x & 15) << 3; // 0 - 15 * 8
  int load_gmem_a_row = by * BM + load_smem_a_row;
  int load_gmem_b_col = bx * BN + load_smem_b_col;

  constexpr int sAoffset = BM * BK;
  constexpr int sBoffset = BK * BN;

  if (load_gmem_a_row >= M || load_gmem_b_col >= N) return;

  uint32_t smemA_base_ptr = __cvta_generic_to_shared(&smemA);
  uint32_t smemB_base_ptr = __cvta_generic_to_shared(&smemB);

  uint32_t RC[4][4][2];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }
  // pre-fetch
  #pragma unroll
  for (int turn = 0; turn < KStage - 1; turn++) {
    int load_gmem_a_col = turn * BK + load_smem_a_col;
    int load_gmem_b_row = turn * BK + load_smem_b_row;
    int load_gmem_a_addr = load_gmem_a_row * K + load_gmem_a_col;
    int load_gmem_b_addr = load_gmem_b_row * N + load_gmem_b_col;
    uint32_t smemA_addr = smemA_base_ptr + (turn * sAoffset + load_smem_a_row * BK + load_smem_a_col) * sizeof(half);
    CP_ASYNC_CG(smemA_addr, &A[load_gmem_a_addr], 16); //bytes, CG bypass L1
    uint32_t smemB_addr = smemB_base_ptr + (turn * sBoffset + load_smem_b_row * BN + load_smem_b_col) * sizeof(half);
    CP_ASYNC_CG(smemB_addr, &B[load_gmem_b_addr], 16); //bytes, CG bypass L1
    CP_ASYNC_COMMIT_GROUP();
  }
  CP_ASYNC_WAIT_GROUP(KStage - 2); // 等待到还剩KStage - 2个操作未完成
  __syncthreads();
  for (int bk = KStage - 1; bk < (K + BK - 1) / BK; bk++) {
    // load tileA
    int smem_select_next = bk % KStage; // load
    int smem_select = (bk + 1) % KStage; // calculate
    int load_gmem_a_col = bk * BK + load_smem_a_col;
    int load_gmem_b_row = bk * BK + load_smem_b_row;
    int load_gmem_a_addr = load_gmem_a_row * K + load_gmem_a_col;
    int load_gmem_b_addr = load_gmem_b_row * N + load_gmem_b_col;
    uint32_t smemA_addr = smemA_base_ptr + (smem_select_next * sAoffset + load_smem_a_row * BK + load_smem_a_col) * sizeof(half);
    uint32_t smemB_addr = smemB_base_ptr + (smem_select_next * sBoffset + load_smem_b_row * BN + load_smem_b_col) * sizeof(half);
    CP_ASYNC_CG(smemA_addr, &A[load_gmem_a_addr], 16); //bytes, CG bypass L1
    CP_ASYNC_CG(smemB_addr, &B[load_gmem_b_addr], 16); //bytes, CG bypass L1
    CP_ASYNC_COMMIT_GROUP();
    uint32_t RA[4][4];
    uint32_t RB[4][2];
    // ldmatrixA
    // m 方向2个warp 负责 128行加载 -> 一个warp64行 64 / 4
    // WARP_M = 4 即为RC在M方向的重复次数
    #pragma unroll  
    for (int i = 0; i < 4; i++) { // warp_m * MMA_M * WARP_M
      int lane_load_smem_a_row = warp_m * 16 * 4 + i * 16 + lane_id % 16;
      int lane_load_smem_a_col = (lane_id / 16) * 8;// 0-15 -> 0, 16-31 -> 8
      uint32_t smemA_addr = __cvta_generic_to_shared(&smemA[smem_select][lane_load_smem_a_row][lane_load_smem_a_col]);
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smemA_addr);
    }
    // n 方向 4个warp，128列加载 -> 一个warp 32列加载，其中一个warp一次加载8列
    // 所以一个warp要加载4次
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int lane_load_smem_b_row = lane_id % 16; //每个线程提供一行smem的首地址 
      int lane_load_smem_b_col = warp_n * 32 + i * 8; 
      uint32_t smemB_addr = __cvta_generic_to_shared(&smemB[smem_select][lane_load_smem_b_row][lane_load_smem_b_col]);
      LDMATRIX_X2_T(RB[i][0], RB[i][1], smemB_addr);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
      }
    }
    CP_ASYNC_WAIT_GROUP(KStage - 2);
    __syncthreads();
  }
  //calculate buffer idx 1
  uint32_t RA[4][4];
  uint32_t RB[4][2];
  #pragma unroll  
  for (int i = 0; i < 4; i++) { // warp_m * MMA_M * WARP_M
    int lane_load_smem_a_row = warp_m * 16 * 4 + i * 16 + lane_id % 16;
    int lane_load_smem_a_col = (lane_id / 16) * 8;// 0-15 -> 0, 16-31 -> 8
    uint32_t smemA_addr = __cvta_generic_to_shared(&smemA[1][lane_load_smem_a_row][lane_load_smem_a_col]);
    LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], smemA_addr);
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    int lane_load_smem_b_row = lane_id % 16; //每个线程提供一行smem的首地址 
    int lane_load_smem_b_col = warp_n * 32 + i * 8; 
    uint32_t smemB_addr = __cvta_generic_to_shared(&smemB[1][lane_load_smem_b_row][lane_load_smem_b_col]);
    LDMATRIX_X2_T(RB[i][0], RB[i][1], smemB_addr);
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
    }
  }
  // store tileC
  // tileC 128 * 128
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      // lane_id / 4 -> lane_id 0-32 -> 0-8
      int store_smem_c_row = warp_m * 64 + i * 16 + lane_id / 4; //分为两部分存储，因为RC[2]中的两个元素相隔8行
      int store_smem_c_col = warp_n * 32 + j * 8 + (lane_id % 4) * 2;  // 32bits
      LDST32BITS(smemC[store_smem_c_row + 0][store_smem_c_col]) = LDST32BITS(RC[i][j][0]);
      LDST32BITS(smemC[store_smem_c_row + 8][store_smem_c_col]) = LDST32BITS(RC[i][j][1]);
    }
  }
  for (int tid = threadIdx.x; tid < BM * BN; tid += blockDim.x) {
    int store_smem_c_row = tid / BN;
    int store_smem_c_col = tid % BN;
    int store_gmem_c_row = by * BM + store_smem_c_row;
    int store_gmem_c_col = bx * BN + store_smem_c_col;
    if (store_gmem_c_row < M && store_gmem_c_col < N) {
      C[store_gmem_c_row * N + store_gmem_c_col] += smemC[store_smem_c_row][store_smem_c_col];
    }
  }
}

__global__ void print(const half* data) {
    printf("half data: %f\n", __half2float(data[0]));
    printf("half data: %f\n", __half2float(data[1]));
}

template <>
void MatMulOperator<__nv_bfloat16>::forward(const Tensor<__nv_bfloat16>* A, const Tensor<__nv_bfloat16>* B, Tensor<__nv_bfloat16>* output, Tensor<__nv_bfloat16>* bias) {
    if (A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("Both input tensors must be 2D matrices.");
    }
    int M = A->shape()[0];
    int N = B->shape()[1];
    int K = A->shape()[1];
    static __nv_bfloat16 alpha = 1.0;
    static __nv_bfloat16 beta = 0.0;
    // dim3 block_x(256);
    // dim3 grid_x((n + 127) / 128, (m + 127) / 128);
    // gemm_mma_vectorized_kernel<<<grid_x, block_x, 0, A->getStream()>>>(
    //     A->data_ptr(), B->data_ptr(), output->data_ptr(), m, n, k);

    cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B->data_ptr(),
      CUDA_R_16F, N, A->data_ptr(), CUDA_R_16F, K, &beta, output->data_ptr(), CUDA_R_16F, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
template <typename T>
MatMulOperator<T>::MatMulOperator() {
    // 初始化 cuBLAS 句柄
    cublasCreate(&handle_);
}

template <typename T>
MatMulOperator<T>::~MatMulOperator() {
    if (handle_) {
        cublasDestroy(handle_);
        handle_ = nullptr;
    }
}


template <>
void MatMulOperator<float>::forward(const Tensor<float>* A, const Tensor<float>* B, Tensor<float>* output, Tensor<float>* bias) {
    if (A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("Both input tensors must be 2D matrices.");
    }

    int m = A->shape()[0];
    int n = B->shape()[1];
    int k = A->shape()[1];
    
    cublasSetStream(handle_, A->getStream());
    cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
    
  
    static float alpha = 1.0;
    static float beta = 0.0;
  
    cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B->data_ptr(), CUDA_R_32F,
                 n, A->data_ptr(), CUDA_R_32F, k, &beta, output->data_ptr(), CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
    
}

}

