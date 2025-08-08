from infer.ops.embedding import Embedding, LMHead
from infer.ops.attention import Attention
from infer.ops.norm import RMSNorm
from infer.ops.MLP import MLP, Linear, QKVLinear
from infer.ops.sampler import Sampler
from infer.ops.rope import Rope
from transformers import Qwen3Config
from torch import nn
import torch

import os
from pathlib import Path

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
# 模型通过context管理上下文kvcache等序列关系，在每次计算前set，保证传入的positions是绝对位置
# 以及之后的prefill decode使用的是绝对位置

class QwenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 10,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 100000,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scale = head_dim ** -0.5

        self.qkv_proj = QKVLinear(hidden_size, self.q_size, self.kv_size)
        self.o_proj = Linear(self.q_size, hidden_size, False)
        self.rotary_emb = Rope(
            head_dim,
            self.head_dim,
            max_position,
            rope_theta,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
        self.attn = Attention(
            num_heads,
            head_dim,
            self.scale,
            num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # batch_size, seq_len = hidden_states.shape[:2]
        # print(f"layer_idx: {self.layer_idx}")
        # print(f"batch_size: {batch_size}, seq_len: {seq_len}")  # Debugging line
        # if positions.dim() == 1:
        #     # 从 [S] -> [1, S] -> [B, S]
        #     positions_expanded = positions.unsqueeze(0).expand(batch_size, -1)
        # else:
        #     positions_expanded = positions
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # if self.layer_idx == 0:
        #     save_gpu_tensor_to_cpu(q, "/2023022031/Infer/pt/", "layer0_q_custom")
        #     save_gpu_tensor_to_cpu(k, "/2023022031/Infer/pt/", "layer0_k_custom")
        #     save_gpu_tensor_to_cpu(v, "/2023022031/Infer/pt/", "layer0_v_custom")
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(-1, self.num_heads, self.head_dim)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(-1, self.num_kv_heads, self.head_dim)

        # print(f"layer_idx: {self.layer_idx}")  # Debugging line
        # print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")  # Debugging line
        # if self.layer_idx == 0:
        #     save_gpu_tensor_to_cpu(q.clone(), "/2023022031/Infer/pt/", "layer0_q_custom_norm")
        #     save_gpu_tensor_to_cpu(k.clone(), "/2023022031/Infer/pt/", "layer0_k_custom_norm")
        self.rotary_emb(positions, q, k)
        # save_gpu_tensor_to_cpu(q.transpose(1, 2), "/2023022031/Infer/pt/", "layer0_rope_q_custom")
        # save_gpu_tensor_to_cpu(k.transpose(1, 2), "/2023022031/Infer/pt/", "layer0_rope_k_custom")
        o = self.attn(q, k, v)
        # if self.layer_idx == 0:
        #     save_gpu_tensor_to_cpu(o, "/2023022031/Infer/pt/", "layer0_o_custom")
        # print(f"Output shape before proj: {o.shape}")  # Debugging line
        output = self.o_proj(o)
        return output
        # qkv = self.qkv_proj(hidden_states)
        # q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # # if self.layer_idx == 0:
        # #     qx = q.view(batch_size, -1, self.num_heads, self.head_dim)
        # #     kx = k.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        # #     vx = v.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        # #     save_gpu_tensor_to_cpu(qx, "/home/zzw/Cute-Learning/Infer/pt/", "layer0_q_custom")
        # #     save_gpu_tensor_to_cpu(kx, "/home/zzw/Cute-Learning/Infer/pt/", "layer0_k_custom")
        # #     save_gpu_tensor_to_cpu(vx, "/home/zzw/Cute-Learning/Infer/pt/", "layer0_v_custom")
        # #     pass
        # q_head = q.view(-1, self.num_heads, self.head_dim)
        # # if self.layer_idx == 0:
        # #     q_head_copy = q_head.clone()
        # #     self.q_norm.use_cuda = getattr(self.q_norm, 'use_cuda', False)
        # #     q_head_copy_norm = self.q_norm(q_head_copy)
        # #     self.q_norm.use_cuda = True
        # q_head = self.q_norm(q_head)
        # # if self.layer_idx == 0:
        # #     q_head_copy_norm = q_head_copy_norm.view(batch_size, -1, self.num_heads, self.head_dim)
        # #     save_gpu_tensor_to_cpu(q_head, "/home/zzw/Cute-Learning/Infer/pt/", "layer0_q_custom_norm")
        # #     save_gpu_tensor_to_cpu(q_head_copy_norm, "/home/zzw/Cute-Learning/Infer/pt/", "layer0_q_custom_norm_copy")
        # q = q_head.view(batch_size, -1, self.num_heads, self.head_dim)
        # k_head = k.view(-1, self.num_kv_heads, self.head_dim)
        # k_head = self.k_norm(k_head)
        
        # k = k_head.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        # v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        # # if self.layer_idx == 0:
        # #     # --- 统一的调试和计算逻辑 ---
        # #     # 1. 克隆纯净的输入，用于 native 版本的计算
        # #     q_prerot_for_native = q.clone()
        # #     k_prerot_for_native = k.clone()
        # #     pos_prerot_for_native = positions_expanded.clone()

        # #     # 2. 计算并保存 Native 版本的结果
        # #     # q_native, k_native = self.rotary_emb.forward_native(
        # #     #     pos_prerot_for_native, q_prerot_for_native, k_prerot_for_native
        # #     # )
        # #     # save_gpu_tensor_to_cpu(q_native.transpose(1, 2), "/home/zzw/Cute-Learning/Infer/pt/", "layer0_rope_q_native")
        # #     # save_gpu_tensor_to_cpu(k_native.transpose(1, 2), "/home/zzw/Cute-Learning/Infer/pt/", "layer0_rope_k_native")

        # #     # 3. 在原始 q, k 上执行默认的 (CUDA) RoPE 操作
        # #     #    这个结果将用于后续的 attention 计算和最终的 custom 保存
        # #     q, k = self.rotary_emb(pos_prerot_for_native, q_prerot_for_native, k_prerot_for_native)
            
        # #     # 4. 将刚刚计算出的 CUDA 结果也保存一份，用于对比
        # #     save_gpu_tensor_to_cpu(q.transpose(1, 2), "/home/zzw/Cute-Learning/Infer/pt/", "layer0_rope_q_cuda")
        # #     save_gpu_tensor_to_cpu(k.transpose(1, 2), "/home/zzw/Cute-Learning/Infer/pt/", "layer0_rope_k_cuda")

        # # else:
        # #     # 对于其他层，正常调用 RoPE
        # #     q, k = self.rotary_emb(positions_expanded, q, k)
            
        #     # 使用 CUDA 版本的结果继续后续计算
        #     # q, k = q_cuda, k_cuda
        # # if self.layer_idx == 0:
        # #     kx = k_head.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        # #     qx = q_head.view(batch_size, -1, self.num_heads, self.head_dim)
        # #     save_gpu_tensor_to_cpu(qx, "/home/zzw/Cute-Learning/Infer/pt/", "layer0_q_custom_norm")
        # #     save_gpu_tensor_to_cpu(kx, "/home/zzw/Cute-Learning/Infer/pt/", "layer0_k_custom_norm")
        # # q, k = self.rotary_emb(positions_expanded, q, k)
        # # if self.layer_idx == 0:
        # #     save_gpu_tensor_to_cpu(q.transpose(1, 2), "/home/zzw/Cute-Learning/Infer/pt/", "layer0_rope_q_custom")
        # #     save_gpu_tensor_to_cpu(k.transpose(1, 2), "/home/zzw/Cute-Learning/Infer/pt/", "layer0_rope_k_custom")
        # q, k = self.rotary_emb(positions_expanded, q, k)
        # if self.layer_idx == 0:
        #     save_gpu_tensor_to_cpu(q.transpose(1, 2), "/2023022031/Infer/pt/", "layer0_rope_q_custom")
        #     save_gpu_tensor_to_cpu(k.transpose(1, 2), "/2023022031/Infer/pt/", "layer0_rope_k_custom")
        # o = self.attn(q, k, v)
        # if self.layer_idx == 0:
        #     save_gpu_tensor_to_cpu(o, "/2023022031/Infer/pt/", "layer0_o_custom")

        # o = o.reshape(batch_size, seq_len, -1)
        # output = self.o_proj(o)
        # return output
    
class QwenDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = QwenAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 100000),
            layer_idx=layer_idx,
        )
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # if self.layer_idx == 0:
        #     save_gpu_tensor_to_cpu(hidden_states, "/2023022031/Infer/pt/", "layer0_input_hidden_states_custom")
        #     pass
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
    
class QwenModel(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )
        self.layers = nn.ModuleList([
            QwenDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states    

        
class QwenForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = QwenModel(config)
        self.lm_head = LMHead(
			num_embeddings=config.vocab_size,
			embedding_dim=config.hidden_size
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
    
