#pragma once
#include "operators/operator.hpp"  // 从include根目录引用
#include "tensor.hpp"
#include "operators/opregistry.hpp"

namespace infer {
    __global__ void print(half* data) ;
template <typename T>
class MatMulOperator : public Operator<T> {
public:
    MatMulOperator();
    
    void forward(std::vector<const Tensor<T>*> input0, Tensor<T>* output) override;
    
    OperatorType type() const override { return OperatorType::MATMUL; }
    std::string name() const override { return "MatMul"; }

    void setTransposeA(bool value) { transposeA_ = value; }
    void setTransposeB(bool value) { transposeB_ = value; }
    
private:
    bool transposeA_ = false;
    bool transposeB_ = false;
    cublasHandle_t handle_ = nullptr;
};
REGISTER_OPERATOR(infer::OperatorType::MATMUL, MatMul, infer::MatMulOperator)
} // namespace infer