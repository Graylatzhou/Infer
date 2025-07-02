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
        input_storage[i] = static_cast<float>(input[base_ptr + i * blockDim.x]);
        sum += input_storage[i] * input_storage[i];
    }
    float warp_sum = warpReduce<float, SumOp, 32>(sum);
    if (tid == 0) {
        shared_sum[threadIdx.y] = rsqrtf(warp_sum / dim_size + eps);
    }
    float norm_factor = shared_sum[threadIdx.y];
    for (int i = 0; tid + i * blockDim.x < dim_size; i++) {
        if constexpr (add_bias) {
            output[base_ptr + i * blockDim.x] = static_cast<T>(input_storage[i] * norm_factor 
                * static_cast<float>(weight[tid + i * blockDim.x]) + static_cast<float>(bias[tid + i * blockDim.x]));
        } else {
            output[base_ptr + i * blockDim.x] = static_cast<T>(input_storage[i] * norm_factor 
                * static_cast<float>(weight[tid + i * blockDim.x]));
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
    if (tid == 0) {
        printf("sum = %f", sum);
    }
    int remaining_elements = dim_size - elementPerThread * (threads_Num - 1);
    // 方和
    if (tid < threads_Num - 1) {
#pragma unroll
        for (int i = 0; i < elementPerThread; i++) {
            int index = other_idx * dim_size + tid * elementPerThread + i;
            input_storage[i] = static_cast<float>(input[index]);
            sum += input_storage[i] * input_storage[i];
        }
    } else { //余数处理部分 
        for (int i = 0; i < remaining_elements; i++) {
            int index = other_idx * dim_size + tid * elementPerThread + i;
            input_storage[i] = static_cast<float>(input[index]);
            sum += input_storage[i] * input_storage[i];
        }
    }   

    typedef cub::BlockReduce<float, 1024> BlockReduce;
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
                output[index] = static_cast<T>(input_storage[i] * norm_factor 
                    * static_cast<float>(weight[weight_index]) + static_cast<float>(bias[weight_index]));
            } else {
                output[index] = static_cast<T>(input_storage[i] * norm_factor 
                    * static_cast<float>(weight[weight_index]));
            }
        }
    }
    else { //余数处理部分 
        for (int i = 0; i < remaining_elements; i++) {
            int index = other_idx * dim_size + tid * elementPerThread + i;
            int weight_index = tid * elementPerThread + i;
            if constexpr (add_bias) {
                output[index] = static_cast<T>(input_storage[i] * norm_factor 
                    * static_cast<float>(weight[weight_index]) + static_cast<float>(bias[weight_index]));
            } else {
                output[index] = static_cast<T>(input_storage[i] * norm_factor 
                    * static_cast<float>(weight[weight_index]));
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
        constexpr int BLOCK_SIZE = 1024;
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



namespace infer {
template <typename T>
void AddRMSOperator<T>::forward(const Tensor<T>* input, const Tensor<T>* weight, Tensor<T>* output, const Tensor<T>* bias) {
    const float eps = 1e-6f;
    const int dim_size = static_cast<int>(input->shape()[input->shape().size() - 1]);
    std::cout << "AddRMSNorm input shape: " << input->shape().size() << std::endl;
    std::cout << "AddRMSNorm input shape: " << input->shape()[input->shape().size() - 1] << std::endl;
    for (int i = 0; i < input->shape().size(); ++i) {
        std::cout << "AddRMSNorm input shape: " << input->shape()[i] << std::endl;
    }
    const int other_size = static_cast<int>(input->size() / dim_size);
    const int total_size = static_cast<int>(input->size());
    auto stream = input->getStream();
    std::cout << "AddRMSNorm dim_size: " << dim_size << ", other_size: " << other_size << std::endl;
    if (dim_size < 1024) {
        //warp implementation
        dispatchAddRMSNormKernel(
            input->data_ptr(), weight->data_ptr(), output->data_ptr(),
            eps, dim_size, other_size, true, stream, bias ? bias->data_ptr() : nullptr
        );
    } else {
        // block implementation
        // 32 * 32 threads per block -> 1024 threads per block
        dispatchAddRMSNormKernel(
            input->data_ptr(), weight->data_ptr(), output->data_ptr(),
            eps, dim_size, other_size, false, stream, bias ? bias->data_ptr() : nullptr
        );
    }

} 
}