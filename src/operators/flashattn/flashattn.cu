#include "operators/flashattn.hpp"

template <typename T>
__global__ void flash_attn_cuda_kernelv1(const T* Q, const T* K, const T* V, T* output) {
    
}


namespace infer {
// 单batch
// Q_weight
template <typename T>
void FlashAttnOperator<T>::forward(const Tensor<T>* Q, const Tensor<T>* K, const Tensor<T>* V, Tensor<T>* output) {
    const int batch_size = Q->shape()[0];
    
}
}