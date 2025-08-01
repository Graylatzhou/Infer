import torch
import triton
import triton.language as tl
import numpy as np
import time

# 已有的store kernel
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # print(f"store_kvcache: key shape {key.shape}, value shape {value.shape}, k_cache shape {k_cache.shape}, v_cache shape {v_cache.shape}, slot_mapping shape {slot_mapping.shape}")
    # print(f"store_kvcache: key stride {key.stride()}, value stride {value.stride()}, k_cache stride {k_cache.stride()}, v_cache stride {v_cache.stride()}")
    assert k_cache.stride(0) == D and v_cache.stride(0) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

# 新增load kernel
@triton.jit
def load_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    output_k_ptr,
    output_v_ptr,
    slot_mapping_ptr,
    output_stride,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    
    # 从缓存加载KV值
    key = tl.load(k_cache_ptr + cache_offsets)
    value = tl.load(v_cache_ptr + cache_offsets)
    
    # 存储到输出
    output_offsets = idx * output_stride + tl.arange(0, D)
    tl.store(output_k_ptr + output_offsets, key)
    tl.store(output_v_ptr + output_offsets, value)

def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, num_heads: int, head_dim: int):
    N = slot_mapping.numel()
    D = num_heads * head_dim
    
    output_k = torch.empty((N, num_heads, head_dim), dtype=k_cache.dtype, device=k_cache.device)
    output_v = torch.empty((N, num_heads, head_dim), dtype=v_cache.dtype, device=v_cache.device)
    
    load_kvcache_kernel[(N,)](k_cache, v_cache, output_k, output_v, slot_mapping, output_k.stride(0), D)
    return output_k, output_v

def verify_kvcache_kernels():
    """验证KV缓存相关的triton内核与PyTorch操作的一致性"""
    print("===== 开始验证KV缓存Triton内核 =====")
    
    # 设置参数
    batch_size = 16      # 批次大小
    num_heads = 32       # 注意力头数
    head_dim = 128       # 每个头的维度
    max_seq_len = 1024   # 最大序列长度
    
    # 创建随机输入数据
    torch.manual_seed(42)
    key = torch.randn((batch_size, num_heads, head_dim), dtype=torch.float16, device="cuda")
    value = torch.randn((batch_size, num_heads, head_dim), dtype=torch.float16, device="cuda")
    
    # 创建随机的slot映射
    slots = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int32, device="cuda")
    
    # 创建缓存
    k_cache_triton = torch.zeros((max_seq_len * num_heads * head_dim,), 
                            dtype=torch.float16, device="cuda").view(max_seq_len, num_heads * head_dim)
    v_cache_triton = torch.zeros((max_seq_len * num_heads * head_dim,), 
                            dtype=torch.float16, device="cuda").view(max_seq_len, num_heads * head_dim)
    k_cache_torch = torch.zeros((max_seq_len, num_heads * head_dim), dtype=torch.float16, device="cuda")
    v_cache_torch = torch.zeros((max_seq_len, num_heads * head_dim), dtype=torch.float16, device="cuda")
    
    print("1. 测试store_kvcache kernel")
    
    # 使用Triton内核存储KV值
    store_kvcache(key, value, k_cache_triton, v_cache_triton, slots)
    
    # 使用PyTorch直接操作
    for i in range(batch_size):
        slot = slots[i].item()
        # 重塑key和value以匹配cache格式
        k_flat = key[i].reshape(-1)  # 展平为[num_heads*head_dim]
        v_flat = value[i].reshape(-1)
        # 存储到对应位置
        k_cache_torch[slot] = k_flat
        v_cache_torch[slot] = v_flat
    
    # 检查结果是否一致
    k_max_diff = torch.max(torch.abs(k_cache_triton - k_cache_torch)).item()
    v_max_diff = torch.max(torch.abs(v_cache_triton - v_cache_torch)).item()
    print(f"  K缓存最大差异: {k_max_diff}")
    print(f"  V缓存最大差异: {v_max_diff}")
    print(f"  存储操作一致性: {'通过' if k_max_diff < 1e-5 and v_max_diff < 1e-5 else '失败'}")
    
    print("\n2. 测试load_kvcache kernel")
    
    # 重置缓存并存入新数据
    k_cache = torch.randn((max_seq_len, num_heads * head_dim), dtype=torch.float16, device="cuda")
    v_cache = torch.randn((max_seq_len, num_heads * head_dim), dtype=torch.float16, device="cuda")
    
    # 使用Triton内核加载KV值
    output_k_triton, output_v_triton = load_kvcache(k_cache, v_cache, slots, num_heads, head_dim)
    
    # 使用PyTorch直接操作
    output_k_torch = torch.zeros((batch_size, num_heads, head_dim), dtype=torch.float16, device="cuda")
    output_v_torch = torch.zeros((batch_size, num_heads, head_dim), dtype=torch.float16, device="cuda")
    
    for i in range(batch_size):
        slot = slots[i].item()
        # 从cache读取并重塑为[num_heads, head_dim]
        cache_k = k_cache[slot].reshape(num_heads, head_dim)
        cache_v = v_cache[slot].reshape(num_heads, head_dim)
        # 存储到输出
        output_k_torch[i] = cache_k
        output_v_torch[i] = cache_v
    
    # 检查结果是否一致
    k_max_diff = torch.max(torch.abs(output_k_triton - output_k_torch)).item()
    v_max_diff = torch.max(torch.abs(output_v_triton - output_v_torch)).item()
    print(f"  K输出最大差异: {k_max_diff}")
    print(f"  V输出最大差异: {v_max_diff}")
    print(f"  加载操作一致性: {'通过' if k_max_diff < 1e-5 and v_max_diff < 1e-5 else '失败'}")
    
    print("\n3. 性能测试")
    
    # 更大规模的数据进行性能测试
    large_batch = 64
    large_key = torch.randn((large_batch, num_heads, head_dim), dtype=torch.float16, device="cuda")
    large_value = torch.randn((large_batch, num_heads, head_dim), dtype=torch.float16, device="cuda")
    large_slots = torch.randint(0, max_seq_len, (large_batch,), dtype=torch.int32, device="cuda")
    large_k_cache = torch.zeros((max_seq_len, num_heads * head_dim), dtype=torch.float16, device="cuda")
    large_v_cache = torch.zeros((max_seq_len, num_heads * head_dim), dtype=torch.float16, device="cuda")
    
    # 测试Triton存储性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        store_kvcache(large_key, large_value, large_k_cache, large_v_cache, large_slots)
    torch.cuda.synchronize()
    triton_store_time = (time.time() - start) / 100
    
    # 测试PyTorch存储性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        for i in range(large_batch):
            slot = large_slots[i].item()
            k_flat = large_key[i].reshape(-1)
            v_flat = large_value[i].reshape(-1)
            large_k_cache[slot] = k_flat
            large_v_cache[slot] = v_flat
    torch.cuda.synchronize()
    torch_store_time = (time.time() - start) / 100
    
    print(f"  存储性能 - Triton: {triton_store_time*1000:.3f}ms, PyTorch: {torch_store_time*1000:.3f}ms")
    print(f"  Triton存储速度提升: {torch_store_time/triton_store_time:.2f}x")
    
    # 测试Triton加载性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        output_k_triton, output_v_triton = load_kvcache(large_k_cache, large_v_cache, large_slots, num_heads, head_dim)
    torch.cuda.synchronize()
    triton_load_time = (time.time() - start) / 100
    
    # 测试PyTorch加载性能
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        output_k_torch = torch.zeros((large_batch, num_heads, head_dim), dtype=torch.float16, device="cuda")
        output_v_torch = torch.zeros((large_batch, num_heads, head_dim), dtype=torch.float16, device="cuda")
        for i in range(large_batch):
            slot = large_slots[i].item()
            cache_k = large_k_cache[slot].reshape(num_heads, head_dim)
            cache_v = large_v_cache[slot].reshape(num_heads, head_dim)
            output_k_torch[i] = cache_k
            output_v_torch[i] = cache_v
    torch.cuda.synchronize()
    torch_load_time = (time.time() - start) / 100
    
    print(f"  加载性能 - Triton: {triton_load_time*1000:.3f}ms, PyTorch: {torch_load_time*1000:.3f}ms")
    print(f"  Triton加载速度提升: {torch_load_time/triton_load_time:.2f}x")
    
    print("\n===== 验证完成 =====")

import flash_attn
print("Flash Attention版本:", flash_attn.__version__)
if __name__ == "__main__":
    
    verify_kvcache_kernels()