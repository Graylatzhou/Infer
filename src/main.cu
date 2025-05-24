#include "tensor.hpp"
#include "memorypool.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
