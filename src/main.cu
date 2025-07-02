#include "tensor.hpp"
#include "memorypool.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "opfactory.hpp"
#include "operator.hpp"
#include "Inferop.hpp"
#include "unifiedOp.hpp"

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

int main() {
    using namespace infer;

    constexpr int other_size = 1024; // 使用更清晰的命名
    constexpr int dim_size = 1024;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CudaMemoryPoolManager::getInstance().getBufferPool().initialize();
    auto op = std::make_unique<UnifiedOp<__nv_bfloat16>>();
    OperatorRegistry::getInstance().listRegisteredOperators();

    // --- 修正 Tensor 的 Shape 和大小 ---
    // GPU Tensors
    auto A = Tensor<__nv_bfloat16>::Buffer({other_size, dim_size}, Device::CUDA, stream);
    auto B = Tensor<__nv_bfloat16>::Buffer({dim_size}, Device::CUDA, stream); // weight 是一维的
    auto C = Tensor<__nv_bfloat16>::Buffer({other_size, dim_size}, Device::CUDA, stream);

    // CPU Tensors
    auto A_cpu = Tensor<__nv_bfloat16>::Buffer({other_size, dim_size}, Device::CPU, stream);
    auto B_cpu = Tensor<__nv_bfloat16>::Buffer({dim_size}, Device::CPU, stream); // weight_cpu 是一维的
    auto C_cpu = Tensor<__nv_bfloat16>::Buffer({other_size, dim_size}, Device::CPU, stream);
    auto C_cpu_ref = Tensor<__nv_bfloat16>::Buffer({other_size, dim_size}, Device::CPU, stream);

    A.fill(__float2bfloat16(2));
    B.fill(__float2bfloat16(3));
    C.fill(__float2bfloat16(0));

    op->rms_norm(&A, &B, &C, nullptr);
    cudaDeviceSynchronize(); // 等待 GPU 完成

    initialize_data(A_cpu.data_ptr(), B_cpu.data_ptr(), other_size, 1, dim_size, false);
    
    // 使用正确的 other_size 和 dim_size 调用
    cpu_add_rms_norm(A_cpu.data_ptr(), B_cpu.data_ptr(), C_cpu_ref.data_ptr(), other_size, dim_size, 1e-6f, nullptr);
    cudaStreamSynchronize(stream); 
    CudaMemoryPoolManager::getInstance().getBufferPool().copyAsync(C_cpu.data_ptr(), C.data_ptr(), C.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream);
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
    for (int i = 0; i < other_size * dim_size; i++) {
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
