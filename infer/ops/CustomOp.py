import torch.nn as nn
import torch
from typing import Optional

class CustomOp(nn.Module):
    """
    简化版CustomOp，只支持CUDA和PyTorch原生实现
    """
    
    # 简化的注册表
    op_registry: dict[str, type['CustomOp']] = {}
    
    def __new__(cls, *args, **kwargs):
        # 简化创建逻辑，直接使用当前类
        return super().__new__(cls)
    
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()
    
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)
    
    def forward_native(self, *args, **kwargs):
        """PyTorch原生实现"""
        raise NotImplementedError
        
    def forward_cuda(self, *args, **kwargs):
        """CUDA实现"""
        raise NotImplementedError
    
    def dispatch_forward(self):
        """简化的分发逻辑：只选择CUDA或原生实现"""
        if self.enabled() and torch.cuda.is_available():
            return self.forward_cuda
        else:
            return self.forward_native
    
    @classmethod
    def enabled(cls) -> bool:
        """简化的启用逻辑：默认启用"""
        return True
    
    # 简化的注册装饰器
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in cls.op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            cls.op_registry[name] = op_cls
            return op_cls
        return decorator
    

