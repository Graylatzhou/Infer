#include "tensor.hpp"
#include "memorypool.hpp"
#include "opfactory.hpp"
#include "operator.hpp"
#include "Inferop.hpp"
#include "unifiedOp.hpp"
#include "kvcache.hpp"

namespace infer {

class Layer {
public:
    virtual ~Layer() = default;

    virtual void forward(std::vector<Tensor<__nv_bfloat16>*> inputs, 
                         std::vector<Tensor<__nv_bfloat16>*> outputs) = 0;
};



}