#pragma once
#include "operator.hpp"
#include <unordered_map>
#include <memory>
#include <string>
#include "operator.hpp"
namespace infer {

class OperatorRegistry {
private:
    OperatorRegistry() = default;
    std::unordered_map<std::string, std::unordered_map<OperatorType, std::unordered_map<std::string, std::shared_ptr<Operator>>>> registry_;
public:
    static OperatorRegistry& getInstance() {
        static OperatorRegistry instance;
        return instance;
    }
    OperatorRegistry(const OperatorRegistry&) = delete;
    OperatorRegistry& operator=(const OperatorRegistry&) = delete;

    // 非模板注册
    void registerOperator(OperatorType type, const std::string& name, std::shared_ptr<Operator> op) {
        std::string type_name = typeid(*op).name();
        registry_[type_name][type][name] = op;
    }

    template<typename T, template<typename> class OpClass>
    std::shared_ptr<OpClass<T>> getOperator(OperatorType type, const std::string& name) {
        std::string type_name = typeid(OpClass<T>).name();
        auto typeIt = registry_.find(type_name);
        if (typeIt != registry_.end()) {
            auto opIt = typeIt->second.find(type);
            if (opIt != typeIt->second.end()) {
                auto nameIt = opIt->second.find(name);
                if (nameIt != opIt->second.end()) {
                    return std::dynamic_pointer_cast<OpClass<T>>(nameIt->second);
                }
            }
        }
        return nullptr;
    }
    void listRegisteredOperators() const {
        std::cout << "=== Registered Operators ===" << std::endl;
        for (const auto& [type_name, type_map] : registry_) {
            std::cout << "TypeName: " << type_name << std::endl;
            for (const auto& [op_type, name_map] : type_map) {
                std::cout << "  OperatorType: " << static_cast<int>(op_type) << std::endl;
                for (const auto& [name, ptr] : name_map) {
                    std::cout << "    Name: " << name << std::endl;
                }
            }
        }
    }
};

#define REGISTER_OPERATOR(type, name, classname, data_type) \
    if constexpr (std::is_same_v<data_type, __nv_bfloat16>) { \
        auto instance_bfloat16 = std::make_shared<classname<__nv_bfloat16>>(); \
        static bool _registered_##classname##_##name = [instance_bfloat16]() { \
            infer::OperatorRegistry::getInstance().registerOperator(type, #name, instance_bfloat16); \
            return true; \
        }(); \
    } else if constexpr (std::is_same_v<data_type, float>) { \
        auto instance_float = std::make_shared<classname<float>>(); \
        static bool _registered_##classname##_##name##_float = [instance_float]() { \
            infer::OperatorRegistry::getInstance().registerOperator(type, #name, instance_float); \
            return true; \
        }(); \
    } else { \
        throw std::runtime_error("Unsupported data type for operator registration"); \
    } 

} // namespace infer