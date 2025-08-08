from infer.models.Bert import BertModel
from infer.utils.weight_loader import load_bin_file
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import BertModel as HFBertModel
from typing import Dict

def save_gpu_tensor_to_cpu(tensor: torch.Tensor, save_path: str, tensor_name: str = None) -> str:
    """
    将 GPU 上的 tensor clone、detach 并保存到 CPU 的 .pt 文件
    
    Args:
        tensor: 在 GPU 上的 tensor
        save_path: 保存路径（文件夹或完整文件路径）
        tensor_name: tensor 的名称，用于生成文件名
    
    Returns:
        str: 实际保存的文件路径
    """
    from pathlib import Path
    # 克隆、分离并转移到 CPU
    cpu_tensor = tensor.clone().detach().cpu()
    
    # 处理保存路径
    save_path = Path(save_path)
    
    if save_path.is_dir() or not save_path.suffix:
        # 如果是文件夹或没有后缀，自动生成文件名
        save_path.mkdir(parents=True, exist_ok=True)
        if tensor_name:
            filename = f"{tensor_name}.pt"
        else:
            filename = f"tensor_{cpu_tensor.shape}_{cpu_tensor.dtype}.pt"
        full_path = save_path / filename
    else:
        # 如果是完整文件路径
        save_path.parent.mkdir(parents=True, exist_ok=True)
        full_path = save_path
    
    # 保存到文件
    torch.save(cpu_tensor, full_path)
    
    print(f"Tensor saved: {full_path}")
    print(f"  - Shape: {cpu_tensor.shape}")
    print(f"  - Dtype: {cpu_tensor.dtype}")
    print(f"  - Device: {cpu_tensor.device}")
    
    return str(full_path)

def compare_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor, 
                   name_a: str = "Tensor A", name_b: str = "Tensor B",
                   rtol: float = 1e-5, atol: float = 1e-8) -> Dict:
    """
    详细对比两个 tensors
    
    Args:
        tensor_a, tensor_b: 要对比的 tensors
        name_a, name_b: tensor 的名称
        rtol, atol: 相对和绝对容差
    
    Returns:
        Dict: 包含对比结果的字典
    """
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Comparing: {name_a} vs {name_b}")
    print(f"{'='*60}")
    
    # 基本信息对比
    print(f"Shape:    {name_a}: {tensor_a.shape} | {name_b}: {tensor_b.shape}")
    print(f"Dtype:    {name_a}: {tensor_a.dtype} | {name_b}: {tensor_b.dtype}")
    print(f"Device:   {name_a}: {tensor_a.device} | {name_b}: {tensor_b.device}")
    
    result = {
        'shapes_match': tensor_a.shape == tensor_b.shape,
        'dtypes_match': tensor_a.dtype == tensor_b.dtype,
        'tensors_close': False,
        'max_abs_diff': float('inf'),
        'mean_abs_diff': float('inf'),
        'max_rel_diff': float('inf'),
        'mean_rel_diff': float('inf'),
        'mismatch_ratio': 1.0
    }
    
    if not result['shapes_match']:
        print("❌ Shape mismatch! Cannot perform numerical comparison.")
        return result
    
    # 转换为相同的数据类型进行比较
    if tensor_a.dtype != tensor_b.dtype:
        print("⚠️  Different dtypes, converting to float32 for comparison")
        tensor_a = tensor_a.to(torch.float32)
        tensor_b = tensor_b.to(torch.float32)
    
    # 数值对比
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / (torch.abs(tensor_a) + 1e-8)  # 避免除零
    
    result['max_abs_diff'] = abs_diff.max().item()
    result['mean_abs_diff'] = abs_diff.mean().item()
    result['max_rel_diff'] = rel_diff.max().item()
    result['mean_rel_diff'] = rel_diff.mean().item()
    
    # 使用 torch.allclose 进行比较
    result['tensors_close'] = torch.allclose(tensor_a, tensor_b, rtol=rtol, atol=atol)
    
    # 计算不匹配的元素比例
    close_mask = torch.isclose(tensor_a, tensor_b, rtol=rtol, atol=atol)
    result['mismatch_ratio'] = (~close_mask).float().mean().item()
    
    # 打印结果
    print(f"\nNumerical Comparison:")
    print(f"  Max absolute difference:  {result['max_abs_diff']:.6e}")
    print(f"  Mean absolute difference: {result['mean_abs_diff']:.6e}")
    print(f"  Max relative difference:  {result['max_rel_diff']:.6e}")
    print(f"  Mean relative difference: {result['mean_rel_diff']:.6e}")
    print(f"  Mismatch ratio:          {result['mismatch_ratio']:.6f}")
    print(f"  Tensors close (rtol={rtol}, atol={atol}): {'✅ Yes' if result['tensors_close'] else '❌ No'}")
    
    # 如果不匹配，提供更多信息
    if not result['tensors_close']:
        print(f"\nDetailed Analysis:")
        print(f"  Total elements: {tensor_a.numel()}")
        print(f"  Mismatched elements: {(~close_mask).sum().item()}")
        
        # 找到最大差异的位置
        max_diff_idx = abs_diff.argmax()
        max_diff_pos = np.unravel_index(max_diff_idx.item(), tensor_a.shape)
        print(f"  Max difference at position {max_diff_pos}:")
        print(f"    {name_a}: {tensor_a[max_diff_pos].item():.6e}")
        print(f"    {name_b}: {tensor_b[max_diff_pos].item():.6e}")
        print(f"    Difference: {abs_diff[max_diff_pos].item():.6e}")
    
    return result

tokenizer = AutoTokenizer.from_pretrained("/2023022031/Infer/Bert")

config = AutoConfig.from_pretrained("/2023022031/Infer/Bert")
# weight = torch.nn.Parameter(
#             torch.empty((1024, 1024), dtype=torch.bfloat16, device=torch.device("cuda"))
#         )
model = BertModel(config)
model.to(torch.bfloat16)
load_bin_file(model, "/2023022031/Infer/Bert/pytorch_model.bin")


bert_transformer = HFBertModel(config)
import os
weights_path = os.path.join("/2023022031/Infer/Bert", "pytorch_model.bin")
original_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

# 创建一个新的 state_dict，并移除 'bert.' 前缀
new_state_dict = {}
for key, value in original_state_dict.items():
    if key.startswith("bert."):
        new_key = key[5:]  # 去掉 'bert.' 前缀
        new_state_dict[new_key] = value
    else:
        # 如果有不带前缀的权重，也保留它们
        new_state_dict[key] = value
bert_transformer.load_state_dict(new_state_dict, strict=False)
bert_transformer.eval()
bert_transformer.to(torch.bfloat16)

inputs = tokenizer("我爱北京", return_tensors="pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 确保您的模型处于评估模式

bert_transformer.to(device)

input_ids = inputs.to(device)
print(f"Running on device: {device}")
# 3. 在两个模型上运行推理
print("\n--- Running Inference for Comparison ---")
with torch.no_grad():  # 关闭梯度计算以节省内存和加速

    your_hidden_states = model(input_ids['input_ids'], attention_mask=input_ids['attention_mask'])
    your_hidden_states = your_hidden_states[0][1:-1]

    hf_outputs = bert_transformer(**input_ids, output_hidden_states=True)
    hf_hidden_states = hf_outputs.hidden_states[-3]

print("\n--- Comparing Hidden States ---")
print(f"Your model output shape: {your_hidden_states.dtype}")
print(f"Hugging Face model output shape: {hf_hidden_states.dtype}")
hf_hidden_states = hf_hidden_states.to(your_hidden_states.dtype)  # 确保数据类型一致

if your_hidden_states.shape != hf_hidden_states.shape:
    print("❌ FAIL: Output shapes do not match!")
else:
    are_close = torch.allclose(your_hidden_states, hf_hidden_states, atol=1e-4, rtol=1e-3)
    
    if are_close:
        print("✅ PASS: Hidden states are numerically close!")
    else:
        print("❌ FAIL: Hidden states are NOT numerically close.")
        
        abs_diff = torch.abs(your_hidden_states - hf_hidden_states)
        
        print(f"  - Max absolute difference: {abs_diff.max().item():.6f}")
        print(f"  - Mean absolute difference: {abs_diff.mean().item():.6f}")

print("\n--- Sample Data Inspection (first 5 tokens, first 5 features) ---")
print("Your model output sample:")
print(your_hidden_states[0, :5, :5])
print("\nHugging Face model output sample:")
print(hf_hidden_states[0, :5, :5])
save_gpu_tensor_to_cpu(hf_hidden_states, "/2023022031/Infer/pt/hf_hidden_states.pt", "hidden_states")

compare_tensors(your_hidden_states, hf_hidden_states, "Custom BertModel", "Hugging Face BertModel", rtol=1e-2, atol=1e-2)

