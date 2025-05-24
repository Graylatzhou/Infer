#include "matmul.hpp"

namespace infer {
template <>
MatMulOperator<half>::MatMulOperator() : Operator<half>() {
    cublasCreate(&handle_);
}

template <>
std::vector<Tensor<half>> MatMulOperator<half>::forward(const std::vector<Tensor<half>>& inputs, Tensor<half>& output) {
    if (inputs.size() != 2) {
        throw std::runtime_error("MatMulOperator requires exactly two input tensors.");
    }

    const Tensor<half>& A = inputs[0];
    const Tensor<half>& B = inputs[1];

    // 检查输入张量的形状是否匹配
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::runtime_error("Both input tensors must be 2D matrices.");
    }

    cublasSetStream(handle_, A.getStream());
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);

    int m = A.shape()[0];
    int n = B.shape()[1];
    int k = A.shape()[1];
    cublasOperation_t transA = transposeA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transposeB_ ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t status = cublasHgemm(
        handle_,
        transA,                      
        transB,                      
        m, n, k,                    
        &alpha,
        A.data_ptr(), transposeA_ ? k : m,  
        B.data_ptr(), transposeB_ ? n : k,  
        &beta,
        output.data_ptr(), m                
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasHgemm failed.");
    }

    return {output};
}

template <>
MatMulOperator<float>::MatMulOperator() : Operator<float>() {
    cublasCreate(&handle_);
}

template <>
std::vector<Tensor<float>> MatMulOperator<float>::forward(const std::vector<Tensor<float>>& inputs, Tensor<float>& output) {
    if (inputs.size() != 2) {
        throw std::runtime_error("MatMulOperator requires exactly two input tensors.");
    }

    const Tensor<float>& A = inputs[0];
    const Tensor<float>& B = inputs[1];

    // 检查输入张量的形状是否匹配
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::runtime_error("Both input tensors must be 2D matrices.");
    }

    cublasSetStream(handle_, A.getStream());
    const float alpha = 1.0f;
    const float beta = 0.0f;

    int m = A.shape()[0];
    int n = B.shape()[1];
    int k = A.shape()[1];
    cublasOperation_t transA = transposeA_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = transposeB_ ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasStatus_t status = cublasSgemm(
        handle_,
        transA,                      
        transB,                      
        m, n, k,                    
        &alpha,
        A.data_ptr(), transposeA_ ? k : m,  
        B.data_ptr(), transposeB_ ? n : k,  
        &beta,
        output.data_ptr(), m                
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasHgemm failed.");
    }

    return {output};
}
REGISTER_OPERATOR(OperatorType::MATMUL, MatMul, MatMulOperator);

}
