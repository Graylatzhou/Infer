#include "tensor.hpp"
#include "memorypool.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "opfactory.hpp"
#include "operator.hpp"
#include "Inferop.hpp"

// CPU版本的半精度GEMM
void cpu_hgemm(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K) {
  // 在CPU上使用OpenMP加速计算
  #pragma omp parallel for collapse(2)
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float psum = 0.0f;  // 使用float进行累加以提高精度
      for (int k = 0; k < K; k++) {
        psum += __bfloat162float(A[m * K + k]) * __bfloat162float(B[k * N + n]);
      }
      C[m * N + n] = __float2bfloat16(psum);
    }
  }
}

void cpu_hadd(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int size) {
  // 在CPU上使用OpenMP加速计算
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    C[i] = __hadd(A[i], B[i]);
  }
}

void cpu_silu(const __nv_bfloat16* A, __nv_bfloat16* C, int size) {
  // 在CPU上使用OpenMP加速计算
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    float a = __bfloat162float(A[i]);
    C[i] = __float2bfloat16(a / (1.0f + expf(-a)));
  }
}

// 修改 initialize_data 函数
void initialize_data(__nv_bfloat16* A, __nv_bfloat16* B, int M, int N, int K, bool is_device = true) {
  if (is_device) {
      // 为GPU内存创建临时CPU缓冲区
      __nv_bfloat16* A_host = new __nv_bfloat16[M * K];
      __nv_bfloat16* B_host = new __nv_bfloat16[K * N];
      
      // 在CPU上初始化数据
      for (int i = 0; i < M * K; i++) {
          A_host[i] = __float2bfloat16(2);
      }
      for (int i = 0; i < K * N; i++) {
          B_host[i] = __float2bfloat16(2);
      }
      
      // 将数据从CPU复制到GPU
      cudaMemcpy(A, A_host, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
      cudaMemcpy(B, B_host, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
      
      // 释放临时缓冲区
      delete[] A_host;
      delete[] B_host;
  } else {
      // CPU版本的初始化，直接操作指针
      for (int i = 0; i < M * K; i++) {
          A[i] = __float2bfloat16(2);
      }
      for (int i = 0; i < M * K; i++) {
          B[i] = __float2bfloat16(2);
      }
  }
}

int main() {
    using namespace infer;
    std::cout << "Checking registered operators..." << std::endl;
    infer::OperatorRegistry::getInstance().printRegisteredOperators();
    constexpr int N = 1024;
    constexpr int M = 1024;
    constexpr int K = 1024;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CudaMemoryPoolManager::getInstance().getBufferPool().initialize();

    auto A = Tensor<__nv_bfloat16>::Buffer({M * K}, Device::CUDA, stream);
    auto B = Tensor<__nv_bfloat16>::Buffer({M * K}, Device::CUDA, stream);
    auto C = Tensor<__nv_bfloat16>::Buffer({M * K}, Device::CUDA, stream);
    auto C_cpu = Tensor<__nv_bfloat16>::Buffer({M * K}, Device::CPU, stream);
    auto C_cpu_ref = Tensor<__nv_bfloat16>::Buffer({M * K}, Device::CPU, stream);
    auto A_cpu = Tensor<__nv_bfloat16>::Buffer({M * K}, Device::CPU, stream);
    auto B_cpu = Tensor<__nv_bfloat16>::Buffer({M * K}, Device::CPU, stream);

    A.fill(__float2bfloat16(2));
    B.fill(__float2bfloat16(2));
    C.fill(__float2bfloat16(0));

    auto silukernel = infer::OperatorFactory::create<__nv_bfloat16>(infer::OperatorType::SILU, "Silu");
    // std::vector<const Tensor<__nv_bfloat16>*> inputs = {&A, &B};
    silukernel->forward({&A}, &C);
    initialize_data(A_cpu.data_ptr(), B_cpu.data_ptr(), M, N, K, false);
    cpu_silu(A_cpu.data_ptr(), C_cpu_ref.data_ptr(), A.size());
    CudaMemoryPoolManager::getInstance().getBufferPool().copyAsync(C_cpu.data_ptr(), C.data_ptr(), M * K * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
    // Tensor<__nv_bfloat16> A({M, N}, Device::CUDA, "temporary", stream);
    // Tensor<__nv_bfloat16> B({N, K}, Device::CUDA, "temporary", stream);
    // Tensor<__nv_bfloat16> C({M, N}, Device::CUDA, "temporary", stream);
    // Tensor<half> C_cpu({M, N}, Device::CPU, "temporary");
    // Tensor<half> C_cpu_ref({M, N}, Device::CPU, "temporary");
    // Tensor<half> A_cpu({M, N}, Device::CPU, "temporary");
    // Tensor<half> B_cpu({N, K}, Device::CPU, "temporary");
    // // 注释掉fill调用
    // C.fill(__float2half(0));
    // A.fill(__float2half(2));
    // B.fill(__float2half(2));
    // auto matmulkernel = infer::OperatorFactory::create<half>(infer::OperatorType::MATMUL, "MatMul");
    // std::vector<const Tensor<half>*> inputs = {&A, &B};
    // matmulkernel->forward(inputs, &C);
    // initialize_data(A_cpu.data_ptr(), B_cpu.data_ptr(), M, N, K, false);
    // cpu_hgemm(A_cpu.data_ptr(), B_cpu.data_ptr(), C_cpu_ref.data_ptr(), M, N, K);
    // CudaMemoryPoolManager::getInstance().getTemporaryPool().copyAsync(C_cpu.data_ptr(), C.data_ptr(), M * N * sizeof(half), cudaMemcpyDeviceToHost, stream);
    // 现在可以比较CPU和GPU的结果
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(__bfloat162float(C_cpu.data_ptr()[i]) - __bfloat162float(C_cpu_ref.data_ptr()[i]));
        if (diff > 1e-4) {
            correct = false;
            printf("Error at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", 
                    i, __bfloat162float(C_cpu.data_ptr()[i]), 
                    __bfloat162float(C_cpu_ref.data_ptr()[i]), diff);
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
