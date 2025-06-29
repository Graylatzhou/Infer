#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"
#include "cute/tensor.hpp"

namespace infer {
template <typename T>
class MatMulOperator : public Operator {
public:
    MatMulOperator();
    ~MatMulOperator();
    
    void forward(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* output, Tensor<T>* bias = nullptr);
    
    OperatorType type() const override { return OperatorType::MATMUL; }
    std::string name() const override { return "MatMul"; }

    void setTransposeA(bool value) { transposeA_ = value; }
    void setTransposeB(bool value) { transposeB_ = value; }
    
private:
    bool transposeA_ = false;
    bool transposeB_ = false;
    cublasHandle_t handle_ = nullptr;
    int algorithm_ = 1; 
};
template class infer::MatMulOperator<float>;
template class infer::MatMulOperator<__nv_bfloat16>;
} // namespace infer