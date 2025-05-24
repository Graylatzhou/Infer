#pragma once
#include "operator.hpp"
#include <unordered_map>
#include <functional>

namespace infer {

// 算子创建函数的类型别名模板
template <typename T>
using OperatorCreator = std::function<OperatorPtr<T>()>;

class OperatorRegistry {
private:
    OperatorRegistry() = default;
    
    std::unordered_map<OperatorType, std::unordered_map<std::string, OperatorCreator<float>>> fp32Registry_;
    std::unordered_map<OperatorType, std::unordered_map<std::string, OperatorCreator<half>>> fp16Registry_;

public:
    static OperatorRegistry& getInstance() {
        static OperatorRegistry instance;
        return instance;
    }
    
    OperatorRegistry(const OperatorRegistry&) = delete;
    OperatorRegistry& operator=(const OperatorRegistry&) = delete;
    
    // fp32 op
    void registerFP32Operator(OperatorType type, const std::string& name, OperatorCreator<float> creator) {
        fp32Registry_[type][name] = creator;
    }
    
    // fp16 op
    void registerFP16Operator(OperatorType type, const std::string& name, OperatorCreator<half> creator) {
        fp16Registry_[type][name] = creator;
    }
    
    OperatorCreator<float> getFP32Creator(OperatorType type, const std::string& name) {
        auto typeIt = fp32Registry_.find(type);
        if (typeIt != fp32Registry_.end()) {
            auto nameIt = typeIt->second.find(name);
            if (nameIt != typeIt->second.end()) {
                return nameIt->second;
            }
        }
        return nullptr;
    }

    OperatorCreator<half> getFP16Creator(OperatorType type, const std::string& name) {
        auto typeIt = fp16Registry_.find(type);
        if (typeIt != fp16Registry_.end()) {
            auto nameIt = typeIt->second.find(name);
            if (nameIt != typeIt->second.end()) {
                return nameIt->second;
            }
        }
        return nullptr;
    }

    std::vector<std::pair<OperatorType, std::string>> listOperators() const {
        std::vector<std::pair<OperatorType, std::string>> result;
        for (const auto& [type, map] : fp32Registry_) {
            for (const auto& [name, _] : map) {
                result.push_back({type, name});
            }
        }
        
        return result;
    }
};

#define REGISTER_FP32_OPERATOR(type, name, classname) \
    static bool _registered_fp32_##name = []() { \
        infer::OperatorRegistry::getInstance().registerFP32Operator( \
            type, #name, []() -> infer::OperatorPtr<float> { \
                return std::make_shared<classname<float>>(); \
            }); \
        return true; \
    }();

#define REGISTER_FP16_OPERATOR(type, name, classname) \
    static bool _registered_fp16_##name = []() { \
        infer::OperatorRegistry::getInstance().registerFP16Operator( \
            type, #name, []() -> infer::OperatorPtr<half> { \
                return std::make_shared<classname<half>>(); \
            }); \
        return true; \
    }();

#define REGISTER_OPERATOR(type, name, classname) \
    REGISTER_FP32_OPERATOR(type, name, classname) \
    REGISTER_FP16_OPERATOR(type, name, classname)

} // namespace infer