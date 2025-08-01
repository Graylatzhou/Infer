import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_tensor_from_pt(file_path: str) -> torch.Tensor:
    """从 .pt 文件加载 tensor"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    tensor = torch.load(file_path, map_location='cpu')
    print(f"Loaded tensor from {file_path}")
    print(f"  - Shape: {tensor.shape}")
    print(f"  - Dtype: {tensor.dtype}")
    print(f"  - Device: {tensor.device}")
    return tensor

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

def compare_tensor_files(file_a: str, file_b: str, **kwargs) -> Dict:
    """从文件加载并对比两个 tensors"""
    tensor_a = load_tensor_from_pt(file_a)
    tensor_a = tensor_a.reshape(-1)  # 展平以便对比
    tensor_b = load_tensor_from_pt(file_b)
    tensor_b = tensor_b.reshape(-1)  # 展平以便对比
    print(f"tensor_a[:5] : {tensor_a[:5]}")
    print(f"tensor_b[:5] : {tensor_b[:5]}")

    name_a = Path(file_a).stem
    name_b = Path(file_b).stem
    
    return compare_tensors(tensor_a, tensor_b, name_a, name_b, **kwargs)

def batch_compare_tensors(tensor_dir: str, patterns: List[Tuple[str, str]], **kwargs) -> Dict[str, Dict]:
    """
    批量对比多个 tensor 文件
    
    Args:
        tensor_dir: 包含 tensor 文件的目录
        patterns: [(pattern_a, pattern_b), ...] 文件名模式对
        **kwargs: 传递给 compare_tensors 的参数
    
    Returns:
        Dict: {comparison_name: result_dict}
    """
    tensor_dir = Path(tensor_dir)
    results = {}
    
    for pattern_a, pattern_b in patterns:
        files_a = list(tensor_dir.glob(pattern_a))
        files_b = list(tensor_dir.glob(pattern_b))
        
        if not files_a:
            print(f"⚠️  No files found for pattern: {pattern_a}")
            continue
        if not files_b:
            print(f"⚠️  No files found for pattern: {pattern_b}")
            continue
            
        for file_a in files_a:
            for file_b in files_b:
                comparison_name = f"{file_a.stem}_vs_{file_b.stem}"
                results[comparison_name] = compare_tensor_files(str(file_a), str(file_b), **kwargs)
    
    return results

# 使用示例
def main_compare():
    """主对比函数示例"""
    
    # 单个文件对比
    try:
        # result = compare_tensor_files(
        #     "/2023022031/Infer/pt/layer0_input_hidden_states_custom.pt",
        #     "/2023022031/Infer/pt/layer0_input_hidden_states_offical.pt",
        #     rtol=1e-2,
        #     atol=1e-2
        # )
        # if result['tensors_close']:
        #     print("🎉 Tensors match!")
        # else:
        #     print("⚠️  Tensors don't match, check the differences above.")
        
        # result = compare_tensor_files(
        #     "/2023022031/Infer/pt/layer0_query_states_offical.pt",
        #     "/2023022031/Infer/pt/layer0_q_custom.pt",
        #     rtol=1e-2,
        #     atol=1e-2
        # )
        # if result['tensors_close']:
        #     print("🎉 Tensors match!")
        # else:
        #     print("⚠️  Tensors don't match, check the differences above.")
        
        # result = compare_tensor_files(
        #     "/2023022031/Infer/pt/layer0_k_custom.pt",
        #     "/2023022031/Infer/pt/layer0_key_states_offical.pt",
        #     rtol=1e-2,
        #     atol=1e-2
        # )
        # if result['tensors_close']:
        #     print("🎉 Tensors match!")
        # else:
        #     print("⚠️  Tensors don't match, check the differences above.")

        # result = compare_tensor_files(
        #     "/2023022031/Infer/pt/layer0_v_custom.pt",
        #     "/2023022031/Infer/pt/layer0_value_states_offical.pt",
        #     rtol=1e-2,
        #     atol=1e-2
        # )
        # if result['tensors_close']:
        #     print("🎉 Tensors match!")
        # else:
        #     print("⚠️  Tensors don't match, check the differences above.")
            
        # result = compare_tensor_files(
        #     "/2023022031/Infer/pt/layer0_query_states_norm_offical.pt",
        #     "/2023022031/Infer/pt/layer0_q_custom_norm.pt",
        #     rtol=1e-2,
        #     atol=1e-2
        # )
        # if result['tensors_close']:
        #     print("🎉 Tensors match!")
        # else:
        #     print("⚠️  Tensors don't match, check the differences above.")


        result = compare_tensor_files(
            "/2023022031/Infer/pt/layer0_rope_k_custom.pt",
            "/2023022031/Infer/pt/layer0_key_states_rope_offical.pt",
            rtol=1e-2,
            atol=1e-2
        )
        if result['tensors_close']:
            print("🎉 Tensors match!")
        else:
            print("⚠️  Tensors don't match, check the differences above.")

        result = compare_tensor_files(
            "/2023022031/Infer/pt/layer0_attn_output_offical.pt",
            "/2023022031/Infer/pt/layer0_attn_output_custom.pt",
            rtol=1e-2,
            atol=1e-2
        )
        if result['tensors_close']:
            print("🎉 Tensors match!")
        else:
            print("⚠️  Tensors don't match, check the differences above.")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    
    # 批量对比示例
    # results = batch_compare_tensors(
    #     "debug_tensors",
    #     [
    #         ("*official*hidden_states*.pt", "*custom*hidden_states*.pt"),
    #         ("*official*residual*.pt", "*custom*residual*.pt"),
    #     ]
    # )

if __name__ == "__main__":
    main_compare()
    
	