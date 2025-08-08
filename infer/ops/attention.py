import torch
from infer_ops import flash_attn_prefill
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
import triton
import triton.language as tl
from infer.context import get_context
from typing import Optional
from enum import IntEnum
# kv cache根据每个layer分配[batch_size, num_tokens, num_heads, head_dim]
# 每一层都会有个attention

class AttentionType(IntEnum):
    Encode = 0
    Decode = 1

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
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale,
        num_kv_heads,
        attn_type: AttentionType = AttentionType.Decode,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        if attn_type == AttentionType.Decode:
            self.k_cache = self.v_cache = torch.tensor([])

    def forward_native(
        self, 
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False, attn_mask=None)

    def forward_cuda(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        O = torch.empty_like(Q)
        flash_attn_prefill(
            Q, K, V, O
        )
        return O

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        bert: bool = False,
    ) -> torch.Tensor:
        if bert is not False:
            return self.forward_native(Q, K, V)
        return self.forward_offical_fa_impl(Q, K, V)
    
    def forward_offical_fa_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ):
        O = torch.empty_like(Q)
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_kv_heads, self.head_dim)
        V = V.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(K, V, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                K, V = k_cache, v_cache
            O = flash_attn_varlen_func(Q, K, V,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            O = flash_attn_with_kvcache(Q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        O = O.view(-1, self.num_heads * self.head_dim)
        return O

