#include "operators/mul.hpp"

template <typename T=__nv_bfloat16, int KNumElemPerThread=4>
__global__ void mul_kernel(const T* input1, const T* input2, T* output, int size) {
    using namespace cute;
    int total_vec = size / KNumElemPerThread;
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx * 4 >= size) return;
    Tensor input1_tensor = make_tensor(make_gmem_ptr(input1), make_shape(size));
    Tensor input2_tensor = make_tensor(make_gmem_ptr(input2), make_shape(size));
    Tensor output_tensor = make_tensor(make_gmem_ptr(output), make_shape(size));

    Tensor tI1gI1 = local_tile(input1_tensor, make_shape(Int<KNumElemPerThread>{}), make_coord(tidx));
    Tensor tI2gI2 = local_tile(input2_tensor, make_shape(Int<KNumElemPerThread>{}), make_coord(tidx));
    Tensor tOgO = local_tile(output_tensor, make_shape(Int<KNumElemPerThread>{}), make_coord(tidx));

    Tensor tIrI1 = make_tensor_like(tI1gI1);
    Tensor tIrI2 = make_tensor_like(tI2gI2);
    Tensor tOrO = make_tensor_like(tOgO);

    copy(tI1gI1, tIrI1);
    copy(tI2gI2, tIrI2);
    #pragma unroll
    for (int i = 0; i < KNumElemPerThread; ++i) {
        tOrO(i) = tIrI1(i) * tIrI2(i);
    }
    copy(tOrO, tOgO);
    // size - total_vec * KNumElemPerThread = remaining elements
    if (size % KNumElemPerThread != 0 && tidx == total_vec) {
        int remaining = size - total_vec * KNumElemPerThread;
        #pragma unroll
        if (tidx * KNumElemPerThread < size) {
            for (int i = 0; i < remaining; ++i) {
                output[total_vec * KNumElemPerThread + i] = input1[total_vec * KNumElemPerThread + i] * input2[total_vec * KNumElemPerThread + i];
            }
        }
    } else return;
}

namespace infer{

template <typename T>
void MulOperator<T>::forward(const Tensor<T>* a, const Tensor<T>* b, Tensor<T>* output) {
    int size = a->size();

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    mul_kernel<T, 4><<<numBlocks, blockSize, 0, a->getStream()>>>(a->data_ptr(), b->data_ptr(), output->data_ptr(), size);
}

}