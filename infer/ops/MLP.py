import torch
from infer_ops import matmul, silu_and_mul
from typing import Optional
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

class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.empty((int(out_features), in_features), dtype=dtype, device=torch.device("cuda")))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(int(out_features), dtype=dtype, device=torch.device("cuda")))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward_native(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return F.linear(
            input,
            self.weight,
            self.bias
        )

    def forward_cuda(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        shape = input.shape[:-1] + (self.out_features,)
        out = torch.empty(shape, dtype=self.weight.dtype, device=input.device)
        matmul(
            input,
            self.weight.t(),
            out
        )
        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward_native(input)

    
class QKVLinear(Linear):
    def __init__(
        self,
        hidden_size: int,
        q_size: int,
        kv_size: int,
        bias: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.q_size = q_size
        self.kv_size = kv_size
        super().__init__(hidden_size, q_size + 2 * kv_size, bias=bias)

    @staticmethod
    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, shared_id: str):
        assert shared_id in ["q", "k", "v"]
        if shared_id == "q":
            shared_size = self.q_size
            shared_offset = 0
        elif shared_id == 'k':
            shared_size = self.kv_size  
            shared_offset = self.q_size
        else:
            shared_size = self.kv_size  
            shared_offset = self.q_size + self.kv_size
        '''
        tensor.narrow 是 PyTorch 中用于从张量中提取一个窄条（子集）的方法。
        它沿着指定的维度提取连续的一段数据
        tensor.narrow(dim, start, length)
        '''
        param.data.narrow(0, int(shared_offset), int(shared_size)).copy_(loaded_weight)

class MergeLinear(Linear):
    def __init__(
        self,
        hidden_size: int,
        output_size: list[int],
    ):
        self.output_size = output_size
        super().__init__(hidden_size, sum(output_size), bias=False)

    @staticmethod
    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shared_offset = 0 if loaded_shard_id == 0 else self.output_size[0]
        shared_size = self.output_size[loaded_shard_id]
        param.data.narrow(0, shared_offset, shared_size).copy_(loaded_weight)
        

'''
SwiGLU
1.Gate Projection: sliu(x @ weight_gate)
2.Up Projection: x @ weight_up
3.Element-wise multiplication: gate_projection * up_projection
'''
class MLP(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int, # MLP中间层的维度
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergeLinear(hidden_size, [intermediate_size] * 2)
        self.down_proj = Linear(intermediate_size, hidden_size)
        self.act_fn = silu_and_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        output_shape = x.shape[:-1] + (self.intermediate_size,)
        act_output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.act_fn(act_output, x)
        x = self.down_proj(act_output)
        return x

class TorchMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        act_output = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(act_output)


if __name__ == "__main__":
    # --- 1. 初始化模型和输入 ---
    device = 'cuda'
    dtype = torch.bfloat16
    hidden_size = 512
    intermediate_size = 2048
    batch_size = 10

    custom_mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size).to(device=device, dtype=dtype)
    
    torch_mlp = TorchMLP(hidden_size=hidden_size, intermediate_size=intermediate_size).to(device=device, dtype=dtype)

    torch.manual_seed(0)
    x = torch.randn((batch_size, hidden_size), device=device, dtype=dtype)

    with torch.no_grad():
        # custom_mlp.fc1.weight 是 (intermediate_size * 2, hidden_size)
        # 我们将其拆分给 gate_proj 和 up_proj
        gate_weight, up_weight = custom_mlp.fc1.weight.chunk(2, dim=0)
        
        torch_mlp.gate_proj.weight.copy_(gate_weight)
        torch_mlp.up_proj.weight.copy_(up_weight)
        
        # fc2 对应 down_proj
        torch_mlp.down_proj.weight.copy_(custom_mlp.fc2.weight)

    # --- 3. 执行前向传播 ---
    print("--- Running Custom MLP ---")
    output_custom = custom_mlp(x)

    print("\n--- Running Torch MLP ---")
    output_torch = torch_mlp(x)

    # --- 4. 数值对比 ---
    print("\n--- Comparison ---")
    print(f"Custom MLP output shape: {output_custom.shape}")
    print(f"Torch MLP output shape:  {output_torch.shape}")
    
    debug_tensor_comparison(output_custom, output_torch, atol=2e-2, rtol=1e-2, name="MLP Output Comparison")
