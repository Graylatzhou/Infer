#include "tensor.hpp"
#include "memorypool.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "opfactory.hpp"
#include "operator.hpp"
#include "Inferop.hpp"
#include "unifiedOp.hpp"
#include <cstdarg> 

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
          B[i] = __float2bfloat16(3);
      }
  }
}

void cpu_add_rms_norm(const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output, 
                      int other_size, int dim_size, float eps, const __nv_bfloat16* bias = nullptr) {
    #pragma omp parallel for
    for (int i = 0; i < other_size; ++i) {
        // 1. 计算当前行的平方和
        float ss = 0.0f;
        for (int j = 0; j < dim_size; ++j) {
            float val = __bfloat162float(input[i * dim_size + j]);
            ss += val * val;
        }

        // 2. 计算归一化因子 (1 / RMS)
        float norm_factor = 1.0f / sqrtf(ss / dim_size + eps);
        if (i == 0) {
            printf("RMS for row %d: %f\n", i, norm_factor);
        }
        // 3. 归一化、缩放并可选地添加偏置
        for (int j = 0; j < dim_size; ++j) {
            float val = __bfloat162float(input[i * dim_size + j]);
            float w = __bfloat162float(weight[j]);
            float normalized_val = val * norm_factor;
            float scaled_val = normalized_val * w;

            if (bias) {
                scaled_val += __bfloat162float(bias[j]);
            }
            
            output[i * dim_size + j] = __float2bfloat16(scaled_val);
        }
    }
}
__global__ void test_kernel(__nv_bfloat16* a) {
    printf("output[0] = %f\n", __bfloat162float(a[0]));
}

void validate_tensor_ptr(const char* name, void* expected, void* actual) {
    if (expected != actual) {
        std::cerr << "POINTER CORRUPTION: " << name 
                  << " expected=" << expected 
                  << " actual=" << actual << std::endl;
    }
}

template <typename T>
__global__ void gpu_compare_kernel(const T *x, const T *y, int n,
                                   float threshold, int *count,
                                   float *max_error) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  float v0 = x[idx];
  float v1 = y[idx];

  float diff = fabs(v0 - v1);
  if (diff > threshold) {
    atomicAdd(count, 1);

    // for positive floating point, there int representation is in the same
    // order.
    int int_diff = *((int *)(&diff));
    atomicMax((int *)max_error, int_diff);
  }
}
void printf_fail(const char *fmt, ...) {
  int red = 31;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

void printf_ok(const char *fmt, ...) {
  int red = 32;
  int def = 39;

  printf("\033[%dm", red);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\033[%dm", def);
}

template <typename T>
void gpu_compare(const T *x, const T *y, int n, float threshold) {
  int *num_count;
  float *max_error;
  cudaMalloc(&num_count, sizeof(int));
  cudaMalloc(&max_error, sizeof(float));
  cudaMemset(num_count, 0, sizeof(int));
  cudaMemset(max_error, 0, sizeof(float));

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  gpu_compare_kernel<<<grid, block>>>(x, y, n, threshold, num_count, max_error);
  int num = 0;
  float error = 0;
  cudaMemcpy(&num, num_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&error, max_error, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (num == 0) {
    printf_ok("check ok, max_error = %f\n", error);
  } else {
    float p = (100.f * num) / n;
    printf_fail("===============================\n");
    printf_fail("check fail: diff %.1f%% = %d/%d max_error = %f\n", p, num, n,
                error);
    printf_fail("===============================\n");
  }
}

int main() {
    using namespace infer;

    constexpr int M = 81920;
    constexpr int N = 256;
    constexpr int K = 512;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CudaMemoryManager::getInstance().getBufferPool().initialize();
    auto op = std::make_unique<UnifiedOp<__nv_bfloat16>>();
    OperatorRegistry::getInstance().listRegisteredOperators();

    // GPU Tensors
    auto A = Tensor<__nv_bfloat16>({M, K}, Device::CUDA, stream);
    auto B = Tensor<__nv_bfloat16>({N, K}, Device::CUDA, stream);
    auto C = Tensor<__nv_bfloat16>({M, N}, Device::CUDA, stream);
    auto C_ref = Tensor<__nv_bfloat16>({M, N}, Device::CUDA, stream);

    auto A_cpu = Tensor<__nv_bfloat16>({M, K}, Device::CPU, stream);
    auto B_cpu = Tensor<__nv_bfloat16>({N, K}, Device::CPU, stream);
    auto C_cpu = Tensor<__nv_bfloat16>({M, N}, Device::CPU, stream);
    auto C_cpu_ref = Tensor<__nv_bfloat16>({M, N}, Device::CPU, stream);
    // CPU Tensors
    // auto A_cpu = Tensor<__nv_bfloat16>::Buffer({M, K}, Device::CPU, stream);
    // auto B_cpu = Tensor<__nv_bfloat16>::Buffer({N, K}, Device::CPU, stream); // weight_cpu 是一维的
    // auto C_cpu = Tensor<__nv_bfloat16>::Buffer({M, N}, Device::CPU, stream);
    // auto C_cpu_ref = Tensor<__nv_bfloat16>::Buffer({M, N}, Device::CPU, stream);

    A.fill(__float2bfloat16(1.1));
    B.fill(__float2bfloat16(1.2));
    C.fill(__float2bfloat16(0));

    A_cpu.fill(__float2bfloat16(1.1));
    B_cpu.fill(__float2bfloat16(1.2));
    C_cpu.fill(__float2bfloat16(0));
    C_cpu_ref.fill(__float2bfloat16(0));

    cudaError_t err = cudaGetLastError();
    std::cout << "Executing Cute GEMM..." << std::endl;
    cudaStreamSynchronize(stream);
    std::cout << "Aptr  " << A.void_ptr() << std::endl;
    std::cout << "Bptr  " << B.void_ptr() << std::endl;
    std::cout << "Cptr  " << C.void_ptr() << std::endl;
    std::cout << "C_ref " << C_ref.void_ptr() << std::endl;
    op->matmul(&A, &B, &C);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in MatMulOperator: " + std::string(cudaGetErrorString(err)));
    }
        cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, A.getStream());
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.void_ptr(),
      CUDA_R_16BF, N, A.void_ptr(), CUDA_R_16BF, K, &beta, C_ref.void_ptr(), CUDA_R_16BF, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in MatMulOperator: " + std::string(cudaGetErrorString(err)));
    }
    gpu_compare<__nv_bfloat16>(C.data_ptr(), C_ref.data_ptr(), M * N, 1e-4f);

    
    // cpu_hgemm(A_cpu.data_ptr(), B_cpu.data_ptr(), C_cpu_ref.data_ptr(), M, N, K);
    // CudaMemoryManager::getInstance().getBufferPool().copyAsync(C_cpu.data_ptr(), C.data_ptr(), M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
    // printf("C_cpu[0] = %f\n", __bfloat162float(C_cpu.data_ptr()[0]));
    // // cudaStreamSynchronize(stream);
    // bool correct = true;
    // int b = 0;
    // for (int i = 0; i < M * N; i++) {
      
    //     float diff = fabs(__bfloat162float(C_cpu.data_ptr()[i]) - __bfloat162float(C_cpu_ref.data_ptr()[i]));

    //     if (diff > 1e-4) {
    //         correct = false;
    //         printf("Error at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", 
    //                 i, __bfloat162float(C_cpu.data_ptr()[i]), 
    //                 __bfloat162float(C_cpu_ref.data_ptr()[i]), diff);
    //         b++;
    //         if (b>20) break;
    //     }
    // }
    // if (correct) {
    //     printf("Results are correct!\n");
    // } else {
    //     printf("Results are incorrect!\n");
    // }
    return 0;
}
