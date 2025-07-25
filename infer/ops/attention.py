from CustomOp import CustomOp
import torch
from infer_ops import flash_attn_prefill
from typing import Optional
# kv cache根据每个layer分配[batch_size, num_tokens, num_heads, head_dim]
# 每一层都会有个attention
class Attention(CustomOp):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward_native(
        self, 
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        # torch_implementation
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

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
        
        
        