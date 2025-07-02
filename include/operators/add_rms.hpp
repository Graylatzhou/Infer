#pragma once
#include "../operator.hpp"  // 从include根目录引用
#include "../tensor.hpp"
#include "../opregistry.hpp"

namespace infer {
template <typename T>
class AddRMSOperator : public Operator {
public:
    void forward(const Tensor<T>* input, const Tensor<T>* weight, Tensor<T>* output, const Tensor<T>* bias = nullptr);
    
    OperatorType type() const override { return OperatorType::ADD_RMS_NORM; }
    std::string name() const override { return "ADDRMS"; }
};
template class infer::AddRMSOperator<float>;
template class infer::AddRMSOperator<__nv_bfloat16>;
}
