#include "silu.hpp"

template <typename T=half, int KNumElemPerThread=4>
__global__ void silu_kernel(const T* input1, T* output, int size) {
    using namespace cute;
    int total_vec = size / KNumElemPerThread;
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx * 4 >= size) return;
    Tensor input1_tensor = make_tensor(make_gmem_ptr(input1), make_shape(size));
    Tensor output_tensor = make_tensor(make_gmem_ptr(output), make_shape(size));

    Tensor tI1gI1 = local_tile(input1_tensor, make_shape(Int<KNumElemPerThread>{}), make_coord(tidx));
    Tensor tOgO = local_tile(output_tensor, make_shape(Int<KNumElemPerThread>{}), make_coord(tidx));

    Tensor tIrI1 = make_tensor_like(tI1gI1);
    Tensor tOrO = make_tensor_like(tOgO);

    copy(tI1gI1, tIrI1);
    #pragma unroll
    for (int i = 0; i < KNumElemPerThread; ++i) {
        // float value = __half2float(tIrI1(i));
        // tOrO(i) = static_cast<T>(value / (1.0f + expf(-value)));
        if constexpr (std::is_same_v<T, half>) {
            // For half type, we use __float2half for conversion
            tOrO(i) = tIrI1(i) * (T(1) / __float2half((T(1) + __float2half(__expf(-__half2float(tIrI1(i)))))));
        } else {
            // For float type, we can directly use float operations
            ;
        }
    }
    copy(tOrO, tOgO);
    // size - total_vec * KNumElemPerThread = remaining elements
    if (size % KNumElemPerThread != 0 && tidx == total_vec) {
        int remaining = size - total_vec * KNumElemPerThread;
        #pragma unroll
        if (tidx * KNumElemPerThread < size) {
            for (int i = 0; i < remaining; ++i) {
                if constexpr (std::is_same_v<T, half>) {
                    T value = input1[total_vec * KNumElemPerThread + i];
                    float val_f = __half2float(value);
                    float sigmoid_f = 1.0f / (1.0f + expf(-val_f));
                    output[total_vec * KNumElemPerThread + i] = __float2half(val_f * sigmoid_f);
                } else {
                    ;
                }
            }
        }
    } else return;
}

namespace infer {
template <typename T>
void SiluOperator<T>::forward(std::vector<const Tensor<T>*> input0, Tensor<T>* output) {
    auto input1 = input0[0]->data_ptr();
    int size = input0[0]->size();

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    silu_kernel<T, 4><<<numBlocks, blockSize>>>(input1, output->data_ptr(), size);
}
} // namespace infer

