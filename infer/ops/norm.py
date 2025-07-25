from CustomOp import CustomOp
import torch
from infer_ops import rms_norm
from typing import Optional

class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # ones 初始化意味着输出 = normalize(x) * 1 = normalize(x)
        # 即：RMSNorm 开始时是一个纯归一化操作，不改变  
        # 推理前再加载权重
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    
    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
        
    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        if residual is None:
            return rms_norm(x, self.weight, out, None, self.eps)
        else:
            return rms_norm(x, self.weight, out, residual, self.eps)
        
