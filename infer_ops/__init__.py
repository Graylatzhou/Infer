import os
import glob
import torch

# 获取当前文件所在的目录
_lib_dir = os.path.dirname(__file__)

# 在该目录中查找编译好的 .so 文件
try:
    _lib_path = glob.glob(os.path.join(_lib_dir, "_C*.so"))[0]
    print(f"Found library: {_lib_path}")
except IndexError:
    raise ImportError("Could not find the C++ extension library.")

# 加载 C++ 库
try:
    torch.ops.load_library(_lib_path)
    print("Library loaded successfully")
except Exception as e:
    print(f"Error loading library: {e}")
    raise

# 尝试获取算子
try:
    add = torch.ops._C.add
    rms_norm = torch.ops._C.rms_norm
    rms_norm_vllm = torch.ops._C.rms_norm_vllm
    matmul = torch.ops._C.matmul
    embedding = torch.ops._C.embedding
    flash_attn_prefill = torch.ops._C.flash_attn_prefill
    rope = torch.ops._C.rotary_embedding
    silu = torch.ops._C.silu
    softmax = torch.ops._C.softmax
    silu_and_mul = torch.ops._C.silu_and_mul
except AttributeError as e:
    print(f"Failed to find add operator: {e}")
    raise

__all__ = ["add", 
           "rms_norm", 
           "matmul", "embedding", 
           "flash_attn_prefill", "rope", "silu", 
           "softmax", "rms_norm_vllm", "rms_norm_fused",
           "silu_and_mul"]