#pragma once
#include "opfactory.hpp"
#include "tensor.hpp"
namespace infer {
template <typename T>
class UnifiedOp {
public:
    UnifiedOp() {
        OperatorFactory::registerOperators<T>();
    }

    void rope(const Tensor<T>* input, Tensor<T>* output, const Tensor<T>* sin_table, 
        const Tensor<T>* cos_table, const Tensor<int32_t>* position_ids) {
        auto op = OperatorFactory::getRopeOperator<T>();
        if (op) {
            op->forward(input, output, sin_table, cos_table, position_ids);
        } else {
            throw std::runtime_error("Rope operator not found");
        }
    }

    void silu(const Tensor<T>* input, Tensor<T>* output) {
        auto op = OperatorFactory::getSiluOperator<T>();
        if (op) {
            op->forward(input, output);
        } else {
            throw std::runtime_error("Silu operator not found");
        }
    }

    void softmax(const Tensor<T>* input, Tensor<T>* output, int32_t axis) {
        auto op = OperatorFactory::getSoftmaxOperator<T>();
        if (op) {
            op->forward(input, output, axis);
        } else {
            throw std::runtime_error("Softmax operator not found");
        }
    }

    void matmul(const Tensor<T>* a, const Tensor<T>* b, Tensor<T>* output) {
        auto op = OperatorFactory::getMatMulOperator<T>();
        if (op) {
            op->forward(a, b, output);
        } else {
            throw std::runtime_error("MatMul operator not found");
        }
    }

    void add(const Tensor<T>* a, const Tensor<T>* b, Tensor<T>* output) {
        auto op = OperatorFactory::getAddOperator<T>();
        if (op) {
            op->forward(a, b, output);
        } else {
            throw std::runtime_error("Add operator not found");
        }
    }

    void mul(const Tensor<T>* a, const Tensor<T>* b, Tensor<T>* output) {
        auto op = OperatorFactory::getMulOperator<T>();
        if (op) {
            op->forward(a, b, output);
        } else {
            throw std::runtime_error("Mul operator not found");
        }
    }   

    void rms_norm(const Tensor<T>* input, const Tensor<T>* weight, Tensor<T>* output, 
        const Tensor<T>* bias = nullptr) {
        if (weight == nullptr || weight->data_ptr() == nullptr) {
            throw std::runtime_error("RMS norm weight is null or its data pointer is null");
        }
        auto op = OperatorFactory::getAddRMSOperator<T>();
        if (op) {
            op->forward(input, weight, output, bias);
        } else {
            throw std::runtime_error("AddRMS operator not found");
        }
    }

    void embedding(const Tensor<T>* input, const Tensor<T>* weight, Tensor<T>* output) {
        auto op = OperatorFactory::getEmbeddingOperator<T>();
        if (op) {
            op->forward(input, weight, output);
        } else {
            throw std::runtime_error("Embedding operator not found");
        }
    }
};
}

