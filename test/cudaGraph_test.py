import torch
from torch import nn
import time
from infer.ops.MLP import Linear

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

class simpel_model(nn.Module):
    def __init__(self):
        super().__init__()
        num_layer = 5000  # 减少层数以便更好地分析
        self.blocks = torch.nn.ModuleList([Linear(640, 640) for _ in range(num_layer)])
    
    def forward(self, x, y, z):
        a = torch.matmul(x, y)
        b = torch.matmul(x, z)
        c = torch.add(a, b)
        for block in self.blocks:
            c = block(c)
        return c

class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = {}
        self.graph_output = {}
    
    def capture(self, x, y, z):
        assert self.cuda_graph is None
        
        # 添加NVTX标记以便在Nsight Systems中可视化
        torch.cuda.nvtx.range_push("CUDA_Graph_Capture_Start")
        
        # 预热：确保所有GPU资源都已初始化
        print("预热模型...")
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(x, y, z)
        torch.cuda.synchronize()
        
        # 开始捕获CUDA Graph
        print("开始捕获CUDA Graph...")
        torch.cuda.nvtx.range_push("CUDA_Graph_Capture_Process")
        
        self.cuda_graph = torch.cuda.CUDAGraph()
        
        # 创建静态输入张量（这很重要！）
        static_x = torch.zeros_like(x)
        static_y = torch.zeros_like(y)  
        static_z = torch.zeros_like(z)
        
        # 复制输入数据到静态张量
        static_x.copy_(x)
        static_y.copy_(y)
        static_z.copy_(z)
        
        with torch.cuda.graph(self.cuda_graph):
            static_out = self.model(static_x, static_y, static_z)
            
        torch.cuda.synchronize()
        print("CUDA Graph捕获完成")
        
        torch.cuda.nvtx.range_pop()  # CUDA_Graph_Capture_Process
        torch.cuda.nvtx.range_pop()  # CUDA_Graph_Capture_Start

        # 保存静态张量的引用
        self.graph_input['x'] = static_x
        self.graph_input['y'] = static_y
        self.graph_input['z'] = static_z
        self.graph_output['output'] = static_out
        
    def forward(self, x, y, z):
        torch.cuda.nvtx.range_push("CUDA_Graph_Execution")
        
        # 将新输入复制到静态张量
        self.graph_input['x'].copy_(x)
        self.graph_input['y'].copy_(y)
        self.graph_input['z'].copy_(z)
        
        # 重放图
        self.cuda_graph.replay()
        
        torch.cuda.nvtx.range_pop()
        return self.graph_output['output']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def comprehensive_benchmark():
    
    # 创建模型和输入
    torch.cuda.nvtx.range_push("Model_Initialization")
    model = simpel_model().cuda()
    inp = torch.randn(640, 640).cuda()
    model.eval()
    torch.cuda.nvtx.range_pop()
    
    # 第一阶段：原始模型预热和测试
    torch.cuda.nvtx.range_push("Original_Model_Warmup")
    with torch.no_grad():
        for i in range(10):
            _ = model(x=inp, y=inp, z=inp)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    # 第二阶段：原始模型性能测试
    torch.cuda.nvtx.range_push("Original_Model_Benchmark")
    start_time = time.time()
    with torch.no_grad():
        for i in range(50):
            torch.cuda.nvtx.range_push(f"Original_Inference_{i}")
            result0 = model(x=inp, y=inp, z=inp)
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    original_time = time.time() - start_time
    torch.cuda.nvtx.range_pop()
    
    # 第三阶段：CUDA Graph捕获
    graph_runner = CUDAGraphRunner(model)
    inputs = {"x": inp, "y": inp, "z": inp}
    graph_runner.capture(**inputs)
    
    # 第四阶段：CUDA Graph预热
    torch.cuda.nvtx.range_push("CUDA_Graph_Warmup")
    for i in range(10):
        _ = graph_runner(**inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    
    # 第五阶段：CUDA Graph性能测试
    print("5. CUDA Graph性能测试...")
    torch.cuda.nvtx.range_push("CUDA_Graph_Benchmark")
    start_time = time.time()
    for i in range(50):
        torch.cuda.nvtx.range_push(f"Graph_Inference_{i}")
        result1 = graph_runner(**inputs)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    graph_time = time.time() - start_time
    torch.cuda.nvtx.range_pop()

    debug_tensor_comparison(result0, result1, atol=1e-3, rtol=1e-3, name="CUDA Graph Output")
    
    # 结果输出
    print("\n" + "="*60)
    print("性能测试结果:")
    print(f"原始模型时间: {original_time:.4f}s ({original_time*1000/50:.2f}ms per inference)")
    print(f"CUDA Graph时间: {graph_time:.4f}s ({graph_time*1000/50:.2f}ms per inference)")
    print(f"加速比: {original_time/graph_time:.2f}x")
    print("="*60)


def cleanup_and_exit():
    """程序结束时的清理工作"""
    print("开始程序清理...")
    
    # 1. 清理全局变量
    if 'model' in globals():
        del globals()['model']
    if 'graph_runner' in globals():
        del globals()['graph_runner']
    
    import infer.ops.CustomOp as custom_op
    custom_op.CustomOp.cleanup_registry()
    
    # 3. 强制垃圾回收
    import gc
    collected = gc.collect()
    print(f"垃圾回收清理了 {collected} 个对象")
    
    # 4. 清理CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 5. 尝试释放malloc内存池
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        print("释放了malloc内存池")
    except:
        pass
    
    print("程序清理完成")


if __name__ == "__main__":
    comprehensive_benchmark()
    cleanup_and_exit()