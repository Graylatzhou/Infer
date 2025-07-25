// 整个文件修改为激活函数的实现
#include "operators/silu.hpp"


template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}
// Activation and gating kernel template.

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel_vllm(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel_vllm(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  act_and_mul_kernel<__nv_bfloat16, KERNEL<__nv_bfloat16>, ACT_FIRST>               \
            <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),       \
                                         reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), d);



void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(silu_kernel_vllm, true);
}

template <typename T>
__global__ void silu_kernel(T* output, const T* input, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    float x = static_cast<float>(input[idx]);
    output[idx] = static_cast<T>(x / (1.0f + expf(-x)));
  }
}

void silu_impl(torch::Tensor& output, const torch::Tensor& input) {
  int size = input.numel();
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / blockSize;

  const c10::cuda::OptionalCUDAGuard device_guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  silu_kernel<<<numBlocks, blockSize, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), reinterpret_cast<__nv_bfloat16*>(input.data_ptr()), size);
}

// namespace infer {
// template <typename T>
// void SiluOperator<T>::forward(const Tensor<T>* input, Tensor<T>* output) {
//     int size = input->size();
//     int blockSize = 256;
//     int numBlocks = (size + blockSize - 1) / blockSize;
//     silu_kernel<T><<<numBlocks, blockSize, 0, input->getStream()>>>(output->data_ptr(), input->data_ptr(), size);
// }
// } // namespace infer

