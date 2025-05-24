#pragma once
#include "operators/operator.hpp"
#include "tensor.hpp"
#include "operators/opregistry.hpp"

namespace infer {
    __global__ void print(half* data) ;
template <typename T>
class MatMulOperator : public Operator<T> {
public:
    MatMulOperator();
    
    Tensor<T> forward(const Tensor<T>&input1, const Tensor<T>&input2, Tensor<T>& output);
    
    OperatorType type() const override { return OperatorType::MATMUL; }
    std::string name() const override { return "MatMul"; }

    void setTransposeA(bool value) { transposeA_ = value; }
    void setTransposeB(bool value) { transposeB_ = value; }
    
private:
    bool transposeA_ = false;
    bool transposeB_ = false;
    cublasHandle_t handle_ = nullptr;
};

} // namespace infer