from CustomOp import CustomOp
import torch
from infer_ops import rope
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


@CustomOp.register("rope")
class Rope(CustomOp):
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
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.bfloat16) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
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
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin,
                                            self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        rope(positions, query, key, self.head_size, self.cos_sin_cache, self.is_neox_style)
        return query, key
    
    @classmethod
    def enabled(cls) -> bool:
        # 可以根据需要修改启用逻辑
        return True
    