import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """通过名称字符串从模型中获取模块。"""
    for n, m in model.named_modules():
        if n == name:
            return m
    raise NameError(f"Module {name} not found in model.")

def load_model(model: nn.Module, path: str):
    """
    从safetensors文件加载模型权重，并使用模块特定的加载器。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    print(f"Loading weights from {path}")

    # 创建一个从参数名到模块名的映射，方便快速查找
    param_to_module_name = {p_name: m_name for m_name, m in model.named_modules() for p_name, _ in m.named_parameters(prefix=m_name, recurse=False)}

    for file in glob(os.path.join(path, "*.safetensors")):
        print(f"--> Processing file: {os.path.basename(file)}")
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_tensor = f.get_tensor(weight_name)
                
                # 检查是否是需要特殊处理的打包权重
                is_packed = False
                for packed_key in packed_modules_mapping:
                    if packed_key in weight_name:
                        target_module_key, shard_id = packed_modules_mapping[packed_key]
                        # 将文件中的权重名转换为我们模型中的参数名
                        # e.g., "model.layers.0.mlp.gate_proj.weight" -> "model.layers.0.mlp.gate_up_proj.weight"
                        param_name = weight_name.replace(packed_key, target_module_key)
                        
                        param = model.get_parameter(param_name)
                        module_name = param_to_module_name[param_name]
                        module = get_module_by_name(model, module_name)
                        
                        loader = type(module).weight_loader
                        loader(module, param, loaded_tensor, shard_id)
                        is_packed = True
                        break
                
                if not is_packed:
                    param = model.get_parameter(weight_name)
                    module_name = param_to_module_name[weight_name]
                    module = get_module_by_name(model, module_name)
                    loader = getattr(type(module), "weight_loader", default_weight_loader)
                    loader(param, loaded_tensor)



if __name__ == "__main__":
    from infer.model import QwenForCausalLM
    from transformers import AutoConfig
    import torch
    import gc
    
    # 从一开始就设置默认设备为GPU
    torch.set_default_device('cuda')
    config = AutoConfig.from_pretrained('/home/zzw/Cute-Learning/Infer/qwen3')
    with torch.device('cuda'):
        model = QwenForCausalLM(config)
        print(f"Model created")
        load_model(model, '/home/zzw/Cute-Learning/Infer/qwen3')
    torch.set_default_device('cpu')
    
    # 清理
    if 'model' in locals():
        del model
    torch.cuda.empty_cache()
    gc.collect()


# def inspect_weight_names(model_path: str):
#     """检查模型权重文件中的所有参数名称"""
#     import glob
#     print("=== 模型权重文件中的参数名称 ===")
    
#     # 查找所有 .safetensors 文件
#     safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    
#     all_weight_names = []
    
#     for file_path in safetensor_files:
#         print(f"\n检查文件: {os.path.basename(file_path)}")
#         with safe_open(file_path, "pt", "cpu") as f:
#             for weight_name in f.keys():
#                 all_weight_names.append(weight_name)
#                 print(f"  {weight_name}")
    
#     # 按字母顺序排序并查找相关模式
#     all_weight_names.sort()
    
#     print("\n=== 层归一化相关的参数 ===")
#     for name in all_weight_names:
#         if "norm" in name.lower() or "layernorm" in name.lower():
#             print(f"  {name}")
    
#     print("\n=== 第一层解码器的参数示例 ===")
#     for name in all_weight_names:
#         if "layers.0." in name:  # 查看第0层的参数
#             print(f"  {name}")
            
#     return all_weight_names

# if __name__ == "__main__":
#     model_path = '/home/zzw/Cute-Learning/Infer/qwen3'
#     weight_names = inspect_weight_names(model_path)