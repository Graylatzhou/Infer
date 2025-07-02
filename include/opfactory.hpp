#pragma once
#include "opregistry.hpp"
#include "operator.hpp"
#include "operators/silu.hpp"
#include "operators/softmax.hpp"
#include "operators/rope.hpp"
#include "operators/mul.hpp"
#include "operators/matmul.hpp"
#include "operators/add.hpp"
#include "operators/add_rms.hpp"


namespace infer {
class OperatorFactory {
public:
    template<typename T>
    static void registerOperators() {
        REGISTER_OPERATOR(OperatorType::MATMUL, MATMUL, MatMulOperator, T);
        REGISTER_OPERATOR(OperatorType::SOFTMAX, SOFTMAX, SoftmaxOperator, T);
        REGISTER_OPERATOR(OperatorType::SILU, SILU, SiluOperator, T);
        REGISTER_OPERATOR(OperatorType::ROPE, ROPE, RopeOperator, T);
        REGISTER_OPERATOR(OperatorType::MUL, MUL, MulOperator, T);
        REGISTER_OPERATOR(OperatorType::ADD, ADD, AddOperator, T);
        REGISTER_OPERATOR(OperatorType::ADD_RMS_NORM, ADD_RMS_NORM, AddRMSOperator, T);
    }

    template<typename T>
    static std::shared_ptr<MatMulOperator<T>> getMatMulOperator() {
        return OperatorRegistry::getInstance().getOperator<T, MatMulOperator>(OperatorType::MATMUL, "MATMUL");
    }

    template<typename T>
    static std::shared_ptr<SoftmaxOperator<T>> getSoftmaxOperator() {
        return OperatorRegistry::getInstance().getOperator<T, SoftmaxOperator>(OperatorType::SOFTMAX, "SOFTMAX");
    }

    template<typename T>
    static std::shared_ptr<SiluOperator<T>> getSiluOperator() {
        return OperatorRegistry::getInstance().getOperator<T, SiluOperator>(OperatorType::SILU, "SILU");
    }

    template<typename T>
    static std::shared_ptr<RopeOperator<T>> getRopeOperator() {
        return OperatorRegistry::getInstance().getOperator<T, RopeOperator>(OperatorType::ROPE, "ROPE");
    }   

    template<typename T>
    static std::shared_ptr<MulOperator<T>> getMulOperator() {
        return OperatorRegistry::getInstance().getOperator<T, MulOperator>(OperatorType::MUL, "MUL");
    }   

    template<typename T>
    static std::shared_ptr<AddOperator<T>> getAddOperator() {  
        return OperatorRegistry::getInstance().getOperator<T, AddOperator>(OperatorType::ADD, "ADD");
    }
    
    template<typename T>
    static std::shared_ptr<AddRMSOperator<T>> getAddRMSOperator() {
        return OperatorRegistry::getInstance().getOperator<T, AddRMSOperator>(OperatorType::ADD_RMS_NORM, "ADD_RMS_NORM");
    }

};

}