#include "operators/silu.hpp"

template <typename T>
__global__ void silu_kernel(T* output, const T* input, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    float x = static_cast<float>(input[idx]);
    output[idx] = static_cast<T>(x / (1.0f + expf(-x)));
  }
}
namespace infer {
template <typename T>
void SiluOperator<T>::forward(const Tensor<T>* input, Tensor<T>* output) {
    int size = input->size();
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    silu_kernel<T><<<numBlocks, blockSize>>>(output->data_ptr(), input->data_ptr(), size);
}
} // namespace infer

