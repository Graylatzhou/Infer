#pragma once
#include "opregistry.hpp"

namespace infer {

class OperatorFactory {
public:
    static OperatorPtr<float> createFP32(OperatorType type, const std::string& name) {
        auto creator = OperatorRegistry::getInstance().getFP32Creator(type, name);
        if (creator) {
            return creator();
        }
        return nullptr;
    }
    static OperatorPtr<half> createFP16(OperatorType type, const std::string& name) {
        auto creator = OperatorRegistry::getInstance().getFP16Creator(type, name);
        if (creator) {
            return creator();
        }
        return nullptr;
    }
    
    template <typename T>
    static OperatorPtr<T> create(OperatorType type, const std::string& name) {
        if constexpr (std::is_same_v<T, float>) {
            return createFP32(type, name);
        } else if constexpr (std::is_same_v<T, half>) {
            return createFP16(type, name);
        } else {
            return nullptr;
        }
    }
    
    static std::vector<std::pair<OperatorType, std::string>> listAvailableOperators() {
        return OperatorRegistry::getInstance().listOperators();
    }
};

} 