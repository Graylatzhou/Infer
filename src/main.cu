#include "tensor.hpp"
#include "memorypool.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
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
__global__ void gemm_mma_vectorized_kernel(half *A, half *B, half *C, int M, int N, int K) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  __shared__ half smemA[BM][BK];
  __shared__ half smemB[BK][BN];
  __shared__ half smemC[BM][BN];
  int warp_id = threadIdx.x / 32;
  int warp_m = warp_id % 2;
  int warp_n = warp_id / 2;
  int lane_id = threadIdx.x % 32;
  if (bx == 0 && by == 0 && threadIdx.x == 0) {
    printf("A[1] = %f\n", __half2float(A[1]));
  }

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
    LDST128BITS(smemA[load_smem_a_row][load_smem_a_col]) = LDST128BITS(A[load_gmem_a_addr]);
    LDST128BITS(smemB[load_smem_b_row][load_smem_b_col]) = LDST128BITS(B[load_gmem_b_addr]);
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

// CPU版本的半精度GEMM
void cpu_hgemm(const half* A, const half* B, half* C, int M, int N, int K) {
  // 在CPU上使用OpenMP加速计算
  #pragma omp parallel for collapse(2)
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float psum = 0.0f;  // 使用float进行累加以提高精度
      for (int k = 0; k < K; k++) {
        psum += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
      }
      C[m * N + n] = __float2half(psum);
    }
  }
}

// 修改 initialize_data 函数
void initialize_data(half* A, half* B, int M, int N, int K, bool is_device = true) {
  if (is_device) {
      // 为GPU内存创建临时CPU缓冲区
      half* A_host = new half[M * K];
      half* B_host = new half[K * N];
      
      // 在CPU上初始化数据
      for (int i = 0; i < M * K; i++) {
          A_host[i] = __float2half(2);
      }
      for (int i = 0; i < K * N; i++) {
          B_host[i] = __float2half(2);
      }
      
      // 将数据从CPU复制到GPU
      cudaMemcpy(A, A_host, M * K * sizeof(half), cudaMemcpyHostToDevice);
      cudaMemcpy(B, B_host, K * N * sizeof(half), cudaMemcpyHostToDevice);
      
      // 释放临时缓冲区
      delete[] A_host;
      delete[] B_host;
  } else {
      // CPU版本的初始化，直接操作指针
      for (int i = 0; i < M * K; i++) {
          A[i] = __float2half(2);
      }
      for (int i = 0; i < K * N; i++) {
          B[i] = __float2half(2);
      }
  }
}

int main() {
    constexpr int N = 1024;
    constexpr int M = 1024;
    constexpr int K = 1024;
    CudaMemoryPoolManager::getInstance().getTemporaryPool().initialize();
    Tensor<half> A({M, N}, Device::CUDA, "temporary");
    Tensor<half> B({N, K}, Device::CUDA, "temporary");
    Tensor<half> C({M, N}, Device::CUDA, "temporary");
    Tensor<half> C_cpu({M, N}, Device::CPU, "temporary");
    Tensor<half> C_cpu_ref({M, N}, Device::CPU, "temporary");
    Tensor<half> A_cpu({M, N}, Device::CPU, "temporary");
    Tensor<half> B_cpu({N, K}, Device::CPU, "temporary");
    // matmul
    cudaMemset(C.data_ptr(), 0, M * K * sizeof(half));
    initialize_data(A.data_ptr(), B.data_ptr(), M, N, K);
    std::cout << "initialized A, B" << std::endl;
    constexpr int new_BM = 128;
    constexpr int new_BN = 128;
    dim3 block_x(256);
    dim3 grid_x((N + new_BN - 1) / new_BN, (M + new_BM - 1) / new_BM);
    gemm_mma_vectorized_kernel<<<grid_x, block_x>>>(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K);
    // cpu_hgemm
    initialize_data(A_cpu.data_ptr(), B_cpu.data_ptr(), M, N, K, false);
    cpu_hgemm(A_cpu.data_ptr(), B_cpu.data_ptr(), C_cpu_ref.data_ptr(), M, N, K);
    // copy C to CPU
    CudaMemoryPoolManager::getInstance().getTemporaryPool().copyAsync(C_cpu.data_ptr(), C.data_ptr(), M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // 现在可以比较CPU和GPU的结果
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(__half2float(C_cpu.data_ptr()[i]) - __half2float(C_cpu_ref.data_ptr()[i]));
        if (diff > 1e-3) {
            correct = false;
            printf("Error at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", 
                    i, __half2float(C_cpu.data_ptr()[i]), 
                    __half2float(C_cpu_ref.data_ptr()[i]), diff);
            break;
        }
    }
    if (correct) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect!\n");
    }
    return 0;
}
