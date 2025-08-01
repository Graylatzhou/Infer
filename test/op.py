import infer_ops
import torch
import torch.nn.functional as F

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

def add_test():
    a = torch.randn(3, 4, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(3, 4, device='cuda', dtype=torch.bfloat16)
    c = torch.empty_like(a)

    # 使用 infer_ops 中的 add 算子
    infer_ops.add(a, b, c)

    # 使用 PyTorch 的内置加法验证结果
    c1 = a + b

    # 调试比较
    is_pass = debug_tensor_comparison(c, c1, atol=1e-2, rtol=1e-2, name="Add Operation")
    
    if is_pass:
        print("Add test passed: The outputs match!")
    else:
        print("Add test failed: The outputs do not match!")

'''
Input tensor dim_size: 128, other_size: 384
Input tensor dim_size: 128, other_size: 96
Input tensor dim_size: 2560, other_size: 12
'''
def test_realistic_data_ranges():
    """使用随机偶数维度的形状测试 RMS Norm 20 次，包含 bias 测试"""
    
    generated_shapes = set()
    num_tests = 20
    passed_count = 0

    print(f"\n=== 运行 {num_tests} 次随机形状 RMS Norm 测试 ===")
    import random
    for i in range(num_tests):
        # 1. 生成不重复的随机偶数维度
        while True:
            # 生成一些典型的偶数维度
            batch_size = random.randint(1, 4) * 2
            seq_len = random.randint(1, 64) * 2
            num_heads = random.randint(1, 16) * 2
            hidden_size = random.randint(16, 512) * 2
            
            shape = (batch_size, seq_len, num_heads, hidden_size)
            if shape not in generated_shapes:
                generated_shapes.add(shape)
                break
        
        print(f"\n--- Test {i+1}/{num_tests}: Shape={shape} ---")

        # 2. 创建张量
        input_tensor = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 2.0
        input_tensor = torch.clamp(input_tensor, -2.0, 2.0)
        
        weight = torch.normal(mean=1.0, std=0.3, size=(hidden_size,), device='cuda', dtype=torch.bfloat16)
        weight = torch.clamp(weight, 0.1, 3.0)
        
        bias = torch.normal(mean=0.0, std=0.1, size=(hidden_size,), device='cuda', dtype=torch.bfloat16)
        bias = torch.clamp(bias, -0.5, 0.5)
        
        output = torch.empty_like(input_tensor)
        eps = 1e-6

        # 3. 参考实现 (使用 float32 以获得更精确的参考值)
        def reference_rms_norm_with_bias(x, weight, bias=None, eps=1e-6):
            x_f32 = x.to(torch.float32)
            variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_f32 * torch.rsqrt(variance + eps)
            result = x_normed.to(x.dtype) * weight
            if bias is not None:
                result = result + bias
            return result
        
        expected_output_with_bias = reference_rms_norm_with_bias(input_tensor, weight, bias, eps)

        # 4. 调用 CUDA op
        try:
            infer_ops.rms_norm(input_tensor, weight, output, bias, eps)
        except Exception as e:
            print(f"❌ Test {i+1} FAILED on call: {e}")
            continue

        # 5. 对比结果
        is_pass_with_bias = debug_tensor_comparison(
            output, expected_output_with_bias, 
            atol=1e-2, rtol=1e-2, 
            name=f"RMS Norm with Bias (Shape: {shape})"
        )
        
        if is_pass_with_bias:
            passed_count += 1

    print(f"\n{'='*60}")
    print(f"随机形状 RMS Norm 测试总结: {passed_count}/{num_tests} 通过.")
    print(f"{'='*60}")



def simple_rms_norm_test():
    """简化版本的 RMS Norm 测试，便于调试"""
    # 使用更小的张量进行调试
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device='cuda', dtype=torch.bfloat16)
    weight = torch.ones(4, device='cuda', dtype=torch.bfloat16)
    output = torch.empty_like(x)
    eps = 1e-6
    
    print(f"Input: {x}")
    print(f"Weight: {weight}")
    
    # 调用您的实现
    infer_ops.rms_norm(x, weight, output, None, eps)
    
    # 手动计算期望结果
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps) * weight
    
    print(f"Your output: {output}")
    print(f"Expected: {expected}")
    
    # 详细调试
    is_pass = debug_tensor_comparison(output, expected, atol=1e-2, rtol=1e-2, name="Simple RMS Norm")
    
    if is_pass:
        print("Simple RMS Norm test passed!")
    else:
        print("Simple RMS Norm test failed!")


def matmul_test():
    a = torch.randn((32 * 32, 4096), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device='cuda', dtype=torch.bfloat16)
    c = torch.empty((32 * 32, 4096), device='cuda', dtype=torch.bfloat16)
    c1 = torch.empty_like(c)

    # 使用 infer_ops 中的 matmul 算子
    # infer_ops.matmul(a, b, c)
    # plan.run()

    # 使用 PyTorch 的内置 matmul 验证结果
    c1 = a @ b.T

    # 调试比较
    is_pass = debug_tensor_comparison(c, c1, atol=1e-3, rtol=1e-3, name="MatMul Operation")

    if is_pass:
        print("MatMul test passed: The outputs match!")
    else:
        print("MatMul test failed: The outputs do not match!")

def embedding_test():
    indices = torch.tensor([1, 25, 5, 10, 7], device='cuda', dtype=torch.int64)
    weight = torch.randn((32, 4096), device='cuda', dtype=torch.bfloat16)
    output = torch.empty((5, 4096), device='cuda', dtype=torch.bfloat16)

    print(f"Input indices: {indices}")
    print(f"Weight shape: {weight.shape}")

    # 调用您的实现
    infer_ops.embedding(output, indices, weight)

    # 手动计算期望结果
    expected = weight[indices]

    print(f"Your output: {output}")
    print(f"Expected: {expected}")

    # 详细调试
    is_pass = debug_tensor_comparison(output, expected, atol=1e-2, rtol=1e-2, name="Embedding Operation")

    if is_pass:
        print("Embedding test passed!")
    else:
        print("Embedding test failed!")


def flash_attn_prefill_test():
    
    Q = torch.randn((1, 32, 64, 128), device='cuda', dtype=torch.bfloat16)
    K = torch.randn((1, 32, 64, 128), device='cuda', dtype=torch.bfloat16)
    V = torch.randn((1, 32, 64, 128), device='cuda', dtype=torch.bfloat16)
    O = torch.empty((1, 32, 64, 128), device='cuda', dtype=torch.bfloat16)

    infer_ops.flash_attn_prefill(Q, K, V, O)

    def reference_flash_attn_prefill(Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    expected_O = reference_flash_attn_prefill(Q, K, V)

    is_pass = debug_tensor_comparison(O, expected_O, atol=2e-2, rtol=1e-3, name="Flash Attention Prefill")
    if is_pass:
        print("Flash Attention Prefill test passed!")
    else:
        print("Flash Attention Prefill test failed!")

def rope_test():
    batch_size, seq_len, dim = 1, 32, 128
    q_tensor = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)
    k_tensor = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, device='cuda', dtype=torch.int64).unsqueeze(0)

    def _compute_inv_freq(base) -> torch.Tensor:
        inv_freq = 1.0 / (base**(torch.arange(0, 128, 2, dtype=torch.bfloat16) / 128)).to('cuda')
        return inv_freq

    def _compute_cos_sin_cache() -> torch.Tensor:
        inv_freq = _compute_inv_freq(10000)
        t = torch.arange(40960, dtype=torch.bfloat16, device='cuda')
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache.to('cuda')
    
    cache = _compute_cos_sin_cache()
    
    # ✅ 重要：保存原始张量用于参考计算
    original_q = q_tensor.clone()
    original_k = k_tensor.clone()
    
    print(f"=== RoPE 测试开始 ===")
    print(f"输入形状: Q={q_tensor.shape}, K={k_tensor.shape}")
    print(f"Position IDs: {position_ids.shape}")
    print(f"Cache 形状: {cache.shape}")
    
    # 调用 CUDA 实现
    try:
        infer_ops.rope(position_ids, q_tensor, k_tensor, 128, cache, is_neox=False)
        print(f"CUDA 调用成功")
        print(f"输出形状: Q={q_tensor.shape}, K={k_tensor.shape}")
    except Exception as e:
        print(f"CUDA 调用失败: {e}")
        return False
    
    from typing import Optional
    
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
        
    def forward_native(
        positions: torch.Tensor,
        query: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, 128)
        query_rot = query[..., :128]
        query_pass = query[..., 128:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin, False)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, 128)
            key_rot = key[..., :128]
            key_pass = key[..., 128:]
            key_rot = _apply_rotary_emb_torch(key_rot, cos, sin, False)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    # ✅ 使用原始张量计算期望结果
    expected_q_tensor, expected_k_tensor = forward_native(
        position_ids, original_q, cache, original_k, None
    )
    
    print(f"期望输出形状: Q={expected_q_tensor.shape}, K={expected_k_tensor.shape}")

    # ✅ 检查形状一致性
    if q_tensor.shape != expected_q_tensor.shape:
        print(f"❌ 形状不匹配: 实际 {q_tensor.shape} vs 期望 {expected_q_tensor.shape}")
        return False
    
    # ✅ 检查 NaN 值
    if torch.isnan(q_tensor).any() or torch.isnan(k_tensor).any():
        print(f"❌ 输出包含 NaN 值")
        print(f"Q tensor NaN count: {torch.isnan(q_tensor).sum().item()}")
        print(f"K tensor NaN count: {torch.isnan(k_tensor).sum().item()}")
        return False
    
    if torch.isnan(expected_q_tensor).any() or torch.isnan(expected_k_tensor).any():
        print(f"❌ 期望输出包含 NaN 值")
        return False

    # 比较结果
    is_pass_q = debug_tensor_comparison(q_tensor, expected_q_tensor, atol=1e-2, rtol=1e-2, name="RoPE Query")
    is_pass_k = debug_tensor_comparison(k_tensor, expected_k_tensor, atol=1e-2, rtol=1e-2, name="RoPE Key")
    
    is_pass = is_pass_q and is_pass_k
    
    if is_pass:
        print("✅ RoPE test passed!")
    else:
        print("❌ RoPE test failed!")
        
        # ✅ 添加详细调试信息
        print(f"\n=== 详细调试信息 ===")
        print(f"原始 Q 统计: min={original_q.min():.6f}, max={original_q.max():.6f}, mean={original_q.mean():.6f}")
        print(f"原始 K 统计: min={original_k.min():.6f}, max={original_k.max():.6f}, mean={original_k.mean():.6f}")
        print(f"输出 Q 统计: min={q_tensor.min():.6f}, max={q_tensor.max():.6f}, mean={q_tensor.mean():.6f}")
        print(f"输出 K 统计: min={k_tensor.min():.6f}, max={k_tensor.max():.6f}, mean={k_tensor.mean():.6f}")
        print(f"期望 Q 统计: min={expected_q_tensor.min():.6f}, max={expected_q_tensor.max():.6f}, mean={expected_q_tensor.mean():.6f}")
        print(f"期望 K 统计: min={expected_k_tensor.min():.6f}, max={expected_k_tensor.max():.6f}, mean={expected_k_tensor.mean():.6f}")
        
    return is_pass

def silu_test():
    input = torch.randn((32, 4096), device='cuda', dtype=torch.bfloat16)
    output = torch.empty_like(input)
    output_ref = torch.empty_like(input)

    # 使用 infer_ops 中的 silu 算子
    infer_ops.silu(output, input)
    # 使用 PyTorch 的内置 silu 验证结果
    output_ref = F.silu(input)

    # 调试比较
    is_pass = debug_tensor_comparison(output, output_ref, atol=1e-3, rtol=1e-3, name="SiLU Operation")  
    if is_pass:
        print("SiLU test passed: The outputs match!")
    else:
        print("SiLU test failed: The outputs do not match!")


if __name__ == "__main__":
    # print("Running Add test...")
    # add_test()
    
    # print("\n" + "="*60)
    # print("Running simple RMS Norm test...")
    # simple_rms_norm_test()
    
    # print("\n" + "="*60)
    # print("Running full RMS Norm test...")
    # test_realistic_data_ranges()

    print("\n" + "="*60)
    print("Running MatMul test...")
    matmul_test()

    # print("\n" + "="*60)
    # print("Running Embedding test...")
    # embedding_test()

    # print("\n" + "="*60)
    # print("Running Flash Attention Prefill test...")
    # flash_attn_prefill_test()

    # print("\n" + "="*60)
    # print("Running RoPE test...")
    # rope_test()

    # print("\n" + "="*60)
    # print("Running SiLU test...")
    # silu_test()

    print("\nAll tests completed!")

