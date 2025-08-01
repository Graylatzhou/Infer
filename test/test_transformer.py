import torch
from transformers import AutoModelForCausalLM, AutoConfig
from infer.model import QwenForCausalLM as CustomQwenForCausalLM
from infer.utils.weight_loader import load_model
import numpy as np

def get_intermediate_output(model, layer_name, model_type="unknown"):
    """使用 hook 捕获指定层的输出"""
    output = None
    def hook(module, input, out):
        nonlocal output
        print(f"\n=== HOOK DEBUG ({model_type}) ===")
        print(f"Layer: {layer_name}")
        print(f"Input type: {type(input)}")
        if isinstance(input, (list, tuple)):
            print(f"Input length: {len(input)}")
            for i, inp in enumerate(input):
                if hasattr(inp, 'shape'):
                    print(f"  Input[{i}] shape: {inp.shape}")
        
        print(f"Output type: {type(out)}")
        if isinstance(out, (list, tuple)):
            print(f"Output length: {len(out)}")
            for i, o in enumerate(out):
                if hasattr(o, 'shape'):
                    print(f"  Output[{i}] shape: {o.shape}")
                    print(f"  Output[{i}] dtype: {o.dtype}")
        elif hasattr(out, 'shape'):
            print(f"Output shape: {out.shape}")
            print(f"Output dtype: {out.dtype}")
        
        # 根据输出类型选择正确的张量
        if isinstance(out, (list, tuple)):
            if len(out) >= 1 and hasattr(out[0], 'shape'):
                output = out[0].detach().cpu()
                print(f"Captured tensor shape: {output.shape}")
            else:
                print("WARNING: out[0] doesn't have shape attribute!")
                output = None
        else:
            output = out.detach().cpu()
            print(f"Captured tensor shape: {output.shape}")
        print("=== HOOK DEBUG END ===\n")
    
    # 在模型中找到目标层并注册hook
    target_layer = dict(model.named_modules())[layer_name]
    handle = target_layer.register_forward_hook(hook)
    return handle, lambda: output

def print_comparison(tensor_a, tensor_b, name):
    """打印两个张量的详细对比结果"""
    print(f"\n--- Comparing: {name} ---")
    if tensor_a is None or tensor_b is None:
        print("One of the tensors is None. Cannot compare.")
        return
    
    print(f"Shape A: {tensor_a.shape}, Shape B: {tensor_b.shape}")
    print(f"Dtype A: {tensor_a.dtype}, Dtype B: {tensor_b.dtype}")
    
    # 统一转换为 float32 进行精确比较
    tensor_a = tensor_a.to(torch.float32)
    tensor_b = tensor_b.to(torch.float32)
    
    abs_diff = torch.abs(tensor_a - tensor_b)
    
    print(f"Max absolute difference: {abs_diff.max().item():.6f}")
    print(f"Mean absolute difference: {abs_diff.mean().item():.6f}")
    
    # 使用一个合理的容差进行比较
    is_close = torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)
    print(f"torch.allclose (atol=1e-3, rtol=1e-3): {'✅ PASS' if is_close else '❌ FAIL'}")

def main():
    model_path = '/2023022031/Infer/Qwen'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    # --- 1. 加载官方模型 ---
    print("Loading official transformers model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    official_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=dtype, 
        trust_remote_code=True
    ).to(device).eval()
    print("Official model loaded.")

    # --- 2. 加载自定义模型 ---
    print("\nLoading custom model...")
    custom_model = CustomQwenForCausalLM(config).to(device).eval()
    load_model(custom_model, model_path)
    print("Custom model loaded and weights populated.")

    # --- 3. 准备输入 ---
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device, dtype=torch.long)
    positions = torch.arange(0, input_ids.shape[1], device=device, dtype=torch.long)
    print(f"Input IDs shape: {input_ids.shape}, Positions shape: {positions.shape}")
    
    # --- 4. 设置 Hook ---
    # # 捕获第一个 Transformer Block (layers.0) 的输出
    # print("Setting up hooks...")
    # official_hook_handle, get_official_layer0_output = get_intermediate_output(
    #     official_model, 'model.layers.0', "OFFICIAL"
    # )
    # custom_hook_handle, get_custom_layer0_output = get_intermediate_output(
    #     custom_model, 'model.layers.0', "CUSTOM"
    # )
    # --- 5. 执行推理 ---
    print("\nRunning inference...")
    with torch.no_grad():
        # # 官方模型推理
        official_outputs = official_model(input_ids=input_ids, output_hidden_states=True)
        official_final_output = official_outputs.hidden_states[-1]
        
        # 自定义模型推理
        custom_final_output = custom_model(input_ids=input_ids, positions=positions)

    # # --- 6. 获取 Hook 捕获的输出并移除 Hook ---
    # official_layer0_output = get_official_layer0_output()
    # custom_layer0_output = get_custom_layer0_output()
    # if official_layer0_output is not None:
    #     print(f"Official layer0 output shape: {official_layer0_output.shape}")
    # if custom_layer0_output is not None:
    #     print(f"Custom layer0 output shape: {custom_layer0_output.shape}")
    
    # official_hook_handle.remove()
    # custom_hook_handle.remove()
    # print("Inference complete. Hooks removed.")

    # # --- 7. 对比结果 ---
    # # 对比第一层输出
    # print_comparison(custom_layer0_output, official_layer0_output, "Layer 0 Hidden States")
    
    # # 对比最终输出 (在最后一层 norm 之前的 hidden_states)
    # print_comparison(custom_final_output.cpu(), official_final_output.cpu(), "Final Hidden States (before final norm)")

if __name__ == "__main__":
    main()