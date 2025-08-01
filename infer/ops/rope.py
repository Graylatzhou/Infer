import torch
from infer_ops import rope
from typing import Optional

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
    # 克隆、分离并转移到 CPU
    from pathlib import Path
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


def _apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

class Rope(torch.nn.Module):
    def __init__(
        self, 
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base   
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: torch.Tensor
        # persistent=False: 不保存到 state_dict，节省存储空间
        # register_buffer 用于注册一个持久化的缓存, 跟着model到指定的device
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.bfloat16, device='cuda') / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq()
        t = torch.arange(self.max_position_embeddings, dtype=torch.bfloat16, device='cuda')
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self, 
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if offsets is not None:
            positions = positions + offsets

        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin,
                                            self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_dim)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = _apply_rotary_emb_torch(key_rot, cos, sin,
                                              self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key
    
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None
    ) -> None:
        rope(positions, query, key, self.head_dim, self.cos_sin_cache, self.is_neox_style)
    
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ):
        self.forward_cuda(positions, query, key)
    

def debug_tensor_comparison(output, expected, atol=1e-3, rtol=1e-3, name="tensor"):
    """
    详细分析两个张量的差异
    
    Args:
        output: 实际输出张量
        expected: 期望输出张量
        atol: 绝对容差
        rtol: 相对容差
        name: 张量名称（用于打印）
    """
    print(f"\n=== {name} 调试信息 ===")
    
    # 基本信息
    print(f"张量形状: {output.shape}")
    print(f"数据类型: {output.dtype}")
    print(f"总元素数: {output.numel()}")
    
    # 计算差异
    abs_diff = torch.abs(output - expected)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-8)  # 避免除零
    
    # 统计信息
    print(f"\n--- 差异统计 ---")
    print(f"绝对误差 - 最大值: {abs_diff.max().item():.6e}")
    print(f"绝对误差 - 平均值: {abs_diff.mean().item():.6e}")
    print(f"绝对误差 - 中位数: {abs_diff.median().item():.6e}")
    print(f"相对误差 - 最大值: {rel_diff.max().item():.6e}")
    print(f"相对误差 - 平均值: {rel_diff.mean().item():.6e}")
    
    # 容差检查
    abs_tolerance_mask = abs_diff > atol
    rel_tolerance_mask = rel_diff > rtol
    combined_mask = abs_tolerance_mask & rel_tolerance_mask
    
    print(f"\n--- 容差分析 (atol={atol:.1e}, rtol={rtol:.1e}) ---")
    print(f"超出绝对容差的元素: {abs_tolerance_mask.sum().item()} / {output.numel()} ({abs_tolerance_mask.float().mean().item()*100:.2f}%)")
    print(f"超出相对容差的元素: {rel_tolerance_mask.sum().item()} / {output.numel()} ({rel_tolerance_mask.float().mean().item()*100:.2f}%)")
    print(f"同时超出两种容差的元素: {combined_mask.sum().item()} / {output.numel()} ({combined_mask.float().mean().item()*100:.2f}%)")
    
    # 误差分布直方图（按数量级）
    print(f"\n--- 绝对误差分布 ---")
    error_ranges = [
        (0, 1e-6, "< 1e-6"),
        (1e-6, 1e-5, "1e-6 ~ 1e-5"),
        (1e-5, 1e-4, "1e-5 ~ 1e-4"),
        (1e-4, 1e-3, "1e-4 ~ 1e-3"),
        (1e-3, 1e-2, "1e-3 ~ 1e-2"),
        (1e-2, 1e-1, "1e-2 ~ 1e-1"),
        (1e-1, float('inf'), "> 1e-1")
    ]
    
    for min_val, max_val, label in error_ranges:
        if max_val == float('inf'):
            mask = abs_diff >= min_val
        else:
            mask = (abs_diff >= min_val) & (abs_diff < max_val)
        count = mask.sum().item()
        percentage = count / output.numel() * 100
        print(f"{label:>12}: {count:>8} 个元素 ({percentage:>5.2f}%)")
    
    # 显示最大误差的位置和值
    if abs_diff.max() > atol:
        max_error_idx = abs_diff.argmax()
        max_error_coords = torch.unravel_index(max_error_idx, output.shape)
        print(f"\n--- 最大误差位置 ---")
        print(f"位置: {max_error_coords}")
        print(f"实际值: {output.flatten()[max_error_idx].item():.6f}")
        print(f"期望值: {expected.flatten()[max_error_idx].item():.6f}")
        print(f"绝对误差: {abs_diff.flatten()[max_error_idx].item():.6e}")
        print(f"相对误差: {rel_diff.flatten()[max_error_idx].item():.6e}")
    
    # 显示一些样本对比
    print(f"\n--- 样本对比 (前10个元素) ---")
    flat_output = output.flatten()
    flat_expected = expected.flatten()
    flat_abs_diff = abs_diff.flatten()
    
    for i in range(min(10, output.numel())):
        print(f"[{i:2d}] 实际: {flat_output[i].item():>10.6f}, "
              f"期望: {flat_expected[i].item():>10.6f}, "
              f"误差: {flat_abs_diff[i].item():>8.2e}")
    
    # 判断是否通过测试
    is_close = torch.allclose(output, expected, atol=atol, rtol=rtol)
    print(f"\n--- 测试结果 ---")
    print(f"torch.allclose 结果: {'✅ PASS' if is_close else '❌ FAIL'}")
    
    return is_close

def test_rope_implementation():
    """
    测试 Rope 模块的 native 和 CUDA 实现，并对比其输出差异。
    模拟 LLM Prefill 阶段的数据。
    """
    print("\n" + "="*60)
    print("=== Running Rope Implementation Test ===")
    print("="*60)

    # 1. 模拟 LLM 中的典型参数
    batch_size = 2
    seq_len = 128
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    max_position_embeddings = 4096
    base = 10000.0
    is_neox_style = True # Qwen/Llama 使用的是 GPT-J 风格
    dtype = torch.bfloat16
    device = 'cuda'

    # 2. 初始化 Rope 模块和 Cache
    try:
        rope_module = Rope(
            head_dim=head_dim,
            rotary_dim=head_dim, # rotary_dim 通常等于 head_dim
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        ).to(device)
        cos_sin_cache = rope_module.cos_sin_cache
        print("✅ Rope module and cache initialized successfully.")
        print(f"Cache 形状: {cos_sin_cache.shape}")
    except Exception as e:
        print(f"❌ Failed to initialize Rope module: {e}")
        return

    # 3. 创建模拟输入数据
    query_states = torch.randn(batch_size, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    key_states = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    positions = torch.arange(seq_len, device=device, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)

    # 调整形状以匹配 RoPE op 的输入
    query_for_rope = query_states.reshape(-1, num_q_heads, head_dim)
    key_for_rope = key_states.reshape(-1, num_kv_heads, head_dim)
    positions_for_rope = positions.reshape(-1)
    
    print("\n--- Input Tensor Shapes ---")
    print(f"Query (for rope): {query_for_rope.shape}")
    print(f"Key (for rope):   {key_for_rope.shape}")
    print(f"Positions (for rope): {positions_for_rope.shape}")

    # ✅ 重要：保存原始张量用于参考计算和 CUDA 调用
    original_q = query_for_rope.clone()
    original_k = key_for_rope.clone()
    
    # 4. 调用 CUDA 实现 (in-place)
    # 直接使用 clone 后的张量进行计算
    q_cuda_output = original_q.clone()
    k_cuda_output = original_k.clone()
    try:
        rope(positions_for_rope, q_cuda_output, k_cuda_output, head_dim, cos_sin_cache, is_neox_style)
        print("\n✅ CUDA forward pass completed.")
    except Exception as e:
        print(f"\n❌ CUDA forward pass failed: {e}")
        return

    # 5. 使用 Native 实现计算期望结果
    # 使用原始张量进行计算，以保证输入纯净
    print("✅ Running Native implementation to get expected result...")
    expected_q, expected_k = rope_module.forward_native(
        positions=positions_for_rope,
        query=original_q,
        key=original_k
    )

    # 6. 检查 NaN 和形状
    if torch.isnan(q_cuda_output).any() or torch.isnan(k_cuda_output).any():
        print("❌ CUDA output contains NaN values!")
        return
    if q_cuda_output.shape != expected_q.shape:
        print(f"❌ Shape mismatch: Actual {q_cuda_output.shape} vs Expected {expected_q.shape}")
        return

    # 7. 对比结果
    print("\n--- Comparing Outputs ---")
    is_pass_q = debug_tensor_comparison(q_cuda_output, expected_q, atol=1e-2, rtol=1e-2, name="RoPE Query")
    is_pass_k = debug_tensor_comparison(k_cuda_output, expected_k, atol=1e-2, rtol=1e-2, name="RoPE Key")
    
    is_pass = is_pass_q and is_pass_k
    
    print("\n" + "="*60)
    if is_pass:
        print("✅✅✅ RoPE Test Passed! ✅✅✅")
    else:
        print("❌❌❌ RoPE Test Failed! ❌❌❌")
    print("="*60)


if __name__ == '__main__':
    # 运行测试
    test_rope_implementation()