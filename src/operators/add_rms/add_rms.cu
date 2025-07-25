#include "operators/add_rms.hpp"
#include <cub/cub.cuh>

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template<>
struct SumOp<half> {
    __device__ __forceinline__ half operator()(const half &a, const half &b) const {
        return __hadd(a, b);
    }
};

template <typename T, template<typename> class ReduceOp, int thread_group_width = 32>
__device__ __forceinline__ T warpReduce(T value) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        value = ReduceOp<T>()(value, __shfl_xor_sync(0xffffffff, value, mask));
    }
    return value;
}

// 如果 rms_norm dim_size < 1024 -> warp implementation
// 如果 rms_norm dim_size >= 1024 -> block implementation
template<typename T, int elementPerThread, bool add_bias=false>
__global__ void rms_norm_kernel_warp_impl(const T* input, const T* weight, T* output, 
    const float eps, const int dim_size, const int other_size, const T* bias = nullptr) {
    int tid = threadIdx.x;
    // input的地址可以解释为 base_ptr + other_idx * dim_size + tid
    // tid * elementPerThread 计算出每个线程处理的元素范围
    int other_idx = blockIdx.x * 32 + threadIdx.y;
    if (other_idx >= other_size) return;
    float input_storage[elementPerThread];
    float sum = 0.0f;
    int base_ptr = other_idx * dim_size + tid;
    __shared__ float shared_sum[32];
    for (int i = 0; tid + i * blockDim.x < dim_size; i++) {
        input_storage[i] = __bfloat162float(input[base_ptr + i * blockDim.x]);
        sum += input_storage[i] * input_storage[i];
    }
    float warp_sum = warpReduce<float, SumOp, 32>(sum);
    if (tid == 0) {
        shared_sum[threadIdx.y] = rsqrtf(warp_sum / dim_size + eps);
    }
    __syncthreads();
    float norm_factor = shared_sum[threadIdx.y];
    for (int i = 0; tid + i * blockDim.x < dim_size; i++) {
        if constexpr (add_bias) {
            output[base_ptr + i * blockDim.x] = __float2bfloat16(input_storage[i] * norm_factor 
                * __bfloat162float(weight[tid + i * blockDim.x]) + __bfloat162float(bias[tid + i * blockDim.x]));
        } else {
            output[base_ptr + i * blockDim.x] = __float2bfloat16(input_storage[i] * norm_factor 
                * __bfloat162float(weight[tid + i * blockDim.x]));
        }
    }
}   

template<typename T, int elementPerThread, bool add_bias=false>
__global__ void rms_norm_kernel_block_impl(const T* input, const T* weight, T* output, 
    const float eps, const int dim_size, const int other_size, const T* bias = nullptr) {
    int tid = threadIdx.x;
    int other_idx = blockIdx.x;
    const int threads_Num = blockDim.x;
    float input_storage[elementPerThread];
    float sum = 0.0f;
    
    int remaining_elements = dim_size - elementPerThread * (threads_Num - 1);
    // 方和
    if (tid < threads_Num - 1) {
#pragma unroll
        for (int i = 0; i < elementPerThread; i++) {
            int index = other_idx * dim_size + tid * elementPerThread + i;
            input_storage[i] = __bfloat162float(input[index]);
            sum += input_storage[i] * input_storage[i];
        }
    } else { //余数处理部分 
        for (int i = 0; i < remaining_elements; i++) {
            int index = other_idx * dim_size + tid * elementPerThread + i;
            input_storage[i] = __bfloat162float(input[index]);
            sum += input_storage[i] * input_storage[i];
        }
    }   
    typedef cub::BlockReduce<float, 512> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_sum;
    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (tid == 0) {
        shared_sum = rsqrtf(block_sum / dim_size + eps);
    }
    __syncthreads();
    float norm_factor = shared_sum;

    if (tid < threads_Num - 1) {
#pragma unroll
        for (int i = 0; i < elementPerThread; i++) {
            int index = other_idx * dim_size + tid * elementPerThread + i;
            int weight_index = tid * elementPerThread + i;
            if constexpr (add_bias) {
                output[index] = __float2bfloat16(input_storage[i] * norm_factor 
                    * __bfloat162float(weight[weight_index]) + __bfloat162float(bias[weight_index]));
            } else {
                output[index] = __float2bfloat16(input_storage[i] * norm_factor 
                    * __bfloat162float(weight[weight_index]));
            }
        }
    }
    else { //余数处理部分 
        for (int i = 0; i < remaining_elements; i++) {
            int index = other_idx * dim_size + tid * elementPerThread + i;
            int weight_index = tid * elementPerThread + i;
            if constexpr (add_bias) {
                output[index] = __float2bfloat16(input_storage[i] * norm_factor 
                    * __bfloat162float(weight[weight_index]) + __bfloat162float(bias[weight_index]));
            } else {
                output[index] = __float2bfloat16(input_storage[i] * norm_factor 
                    * __bfloat162float(weight[weight_index]));
            }
        }
    }
}


template <typename T>
void dispatchAddRMSNormKernel(
    const T* input, const T* weight, T* output,
    const float eps, const int dim_size, const int other_size,
    bool use_warp_impl, cudaStream_t stream, const T* bias = nullptr
) {
    dim3 block, grid;
    int elementPerThread;
    if (use_warp_impl) {
        block = dim3(32, 32);
        grid = dim3((other_size + block.y - 1) / block.y, 1, 1);
        elementPerThread = min((dim_size + 31) / 32, 32);
        std::cout << "Using warp implementation with elementPerThread: " << elementPerThread << std::endl;
        if (bias) {
            if (elementPerThread <= 1) {
                rms_norm_kernel_warp_impl<T, 1, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 2) {
                rms_norm_kernel_warp_impl<T, 2, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 4) {
                rms_norm_kernel_warp_impl<T, 4, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 8) {
                rms_norm_kernel_warp_impl<T, 8, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 16) {
                rms_norm_kernel_warp_impl<T, 16, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else {
                rms_norm_kernel_warp_impl<T, 32, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            }
        } else {
            if (elementPerThread <= 1) {
                rms_norm_kernel_warp_impl<T, 1, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 2) {
                rms_norm_kernel_warp_impl<T, 2, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 4) {
                rms_norm_kernel_warp_impl<T, 4, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 8) {
                rms_norm_kernel_warp_impl<T, 8, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 16) {
                rms_norm_kernel_warp_impl<T, 16, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else {
                rms_norm_kernel_warp_impl<T, 32, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            }
        }
    } else {
        constexpr int BLOCK_SIZE = 512;
        block = dim3(BLOCK_SIZE);
        grid = dim3(other_size);
        elementPerThread = min((dim_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 32);
        std::cout << "elementPerThread: " << elementPerThread << std::endl;
        std::cout << "Block Impl" << std::endl;
        if (bias) {
            if (elementPerThread <= 1) {
                rms_norm_kernel_block_impl<T, 1, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 2) {
                rms_norm_kernel_block_impl<T, 2, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 4) {
                rms_norm_kernel_block_impl<T, 4, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 8) {
                rms_norm_kernel_block_impl<T, 8, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 16) {
                rms_norm_kernel_block_impl<T, 16, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else {
                rms_norm_kernel_block_impl<T, 32, true><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            }
        } else {
            if (elementPerThread <= 1) {
                rms_norm_kernel_block_impl<T, 1, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 2) {
                rms_norm_kernel_block_impl<T, 2, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 4) {
                rms_norm_kernel_block_impl<T, 4, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 8) {
                rms_norm_kernel_block_impl<T, 8, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else if (elementPerThread <= 16) {
                rms_norm_kernel_block_impl<T, 16, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            } else {
                rms_norm_kernel_block_impl<T, 32, false><<<grid, block, 0, stream>>>(input, weight, output, eps, dim_size, other_size, bias);
            }
        }
    }
}

template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))
// CUDA < 12.0 runs into issues with packed type conversion
template <>
struct _typeConvert<c10::Half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static inline float convert(hip_type x) { return __half2float(x); }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __half22float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// CUDA_ARCH < 800 does not have BF16 support
// TODO: Add in ROCm support once public headers handle bf16 maturely
template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static inline float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};
  #endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#endif    // defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >=
          // 12000))

template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  /* Not theoretically necessary that width is a power of 2 but should
     almost always be the case for optimization purposes */
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        T2 temp{data[i], data[i + 1]};
        temp += T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] += other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        T2 temp{data[i], data[i + 1]};
        temp *= T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] *= other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float temp = Converter::convert(data[i]) * scale;
        data[i] = Converter::convert(temp);
      }
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x * z.x + z.y * z.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        result += x * x;
      }
    }
    return result;
  }
};

template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rms_norm_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()), epsilon, num_tokens, hidden_size);
}

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    fused_add_rms_norm_kernel<__nv_bfloat16, 8>                       \
            <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),           \
                                         reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),        \
                                         reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()), epsilon, \
                                         num_tokens, hidden_size);             
  } else {
    fused_add_rms_norm_kernel<__nv_bfloat16, 0>                       \
            <<<grid, block, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),           \
                                         reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),        \
                                         reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()), epsilon, \
                                         num_tokens, hidden_size);             
  }
}



void rms_norm_impl(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    const c10::optional<torch::Tensor>& bias,
    double eps
) {
     // 1. 输入校验
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on a CUDA device");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be on a CUDA device");
    TORCH_CHECK(input.dim() >= 1, "Input tensor must have at least one dimension");
    const auto last_dim_size = input.size(-1);
    TORCH_CHECK(last_dim_size == weight.size(-1), "Input and weight tensors must have the same last dimension size");
    
    // 假设数据类型为 bfloat16，您可以根据需要扩展
    using T = __nv_bfloat16;
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input must be of type BFloat16");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "Weight must be of type BFloat16");

    const T* bias_ptr = nullptr;
    if (bias.has_value()) {
        const auto& bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on a CUDA device");
        TORCH_CHECK(bias_tensor.scalar_type() == torch::kBFloat16, "Bias must be of type BFloat16");
        TORCH_CHECK(bias_tensor.size(-1) == last_dim_size, "Bias and weight must have the same size");
        bias_ptr = static_cast<T*>(bias_tensor.data_ptr());
    }

    // 2. 获取维度信息
    const int dim_size = static_cast<int>(last_dim_size);
    const int other_size = static_cast<int>(input.numel() / dim_size);


    const c10::cuda::OptionalCUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // 5. 决定使用哪种实现并调用分发函数
    bool use_warp_impl = (dim_size < 1024);
    dispatchAddRMSNormKernel<T>(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<float>(eps),
        dim_size,
        other_size,
        use_warp_impl,
        stream,
        bias_ptr
    );
}



// namespace infer {
// template <typename T>
// void AddRMSOperator<T>::forward(const Tensor<T>* input, const Tensor<T>* weight, Tensor<T>* output, const Tensor<T>* bias) {
//     const float eps = 1e-6f;
//     const int dim_size = static_cast<int>(input->shape()[input->shape().size() - 1]);
//     std::cout << "Verifying pointers before kernel launch:" << std::endl;
//     std::cout << "Input ptr: " << input << " (null? " << (input == nullptr) << ")" << std::endl;
//     std::cout << "Weight ptr: " << weight << " (null? " << (weight == nullptr) << ")" << std::endl;
//     std::cout << "Output ptr: " << output << " (null? " << (output == nullptr) << ")" << std::endl;

//     // 如果使用了bias
//     if (bias) {
//         std::cout << "Bias ptr: " << bias << " (null? " << (bias == nullptr) << ")" << std::endl;
//     }
//     const int other_size = static_cast<int>(input->size() / dim_size);
//     const int total_size = static_cast<int>(input->size());
//     auto stream = input->getStream();
//     if (dim_size < 1024) {
//         //warp implementation
//         dispatchAddRMSNormKernel(
//             input->data_ptr(), weight->data_ptr(), output->data_ptr(),
//             eps, dim_size, other_size, true, stream, bias ? bias->data_ptr() : nullptr
//         );
//     } else {
//         // block implementation
//         // 32 * 32 threads per block -> 1024 threads per block
//         std::cout << "Using block implementation" << std::endl;
//         dispatchAddRMSNormKernel(
//             input->data_ptr(), weight->data_ptr(), output->data_ptr(),
//             eps, dim_size, other_size, false, stream, bias ? bias->data_ptr() : nullptr
//         );
//     }

// } 
// }