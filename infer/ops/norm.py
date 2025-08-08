import torch
from infer_ops import rms_norm
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

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        use_cuda: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.use_cuda = False
        self.hidden_size = hidden_size
        self.eps = eps
        # ones 初始化意味着输出 = normalize(x) * 1 = normalize(x)
        # 即：RMSNorm 开始时是一个纯归一化操作，不改变  
        # 推理前再加载权重
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))

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
            rms_norm(x, self.weight, out, None, self.eps)
        else:
            rms_norm(x, self.weight, out, residual, self.eps)
        
        return out if residual is None else (out, residual)
        
    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)

        
