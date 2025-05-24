#pragma once
#include "operators/operator.hpp"
#include "tensor.hpp"
#include "opregistry.hpp"

namespace infer {

template <typename T>
class MatMulOperator : public Operator<T> {
public:
    MatMulOperator() override;
    
    std::vector<Tensor<T>> forward(const std::vector<Tensor<T>>& inputs, Tensor<T>& output) override;
    
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