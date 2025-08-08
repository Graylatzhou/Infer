from typing import Optional, Union
from infer.ops.embedding import Embedding
from infer.ops.MLP import Linear, QKVLinear
from infer.ops.pool import BasePooler, PoolingType
from infer.ops.attention import Attention
from transformers import BertConfig
import torch.nn as nn
import torch

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
    from pathlib import Path
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

class BertEmbedding(nn.Module):

	def __init__(self, config: BertConfig):
		super().__init__()
		self.size = config.hidden_size
		self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)
		self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

		self.register_buffer("position_ids", torch.arange(
			config.max_position_embeddings).unsqueeze(0)
		)
		self.position_embedding_type = config.position_embedding_type
		if self.position_embedding_type != "absolute":
			raise ValueError(
				f"Unsupported position embedding type: {self.position_embedding_type}"
			)
		
	def forward(
		self,
		input_ids: torch.Tensor,
		position_ids: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		input_shape = input_ids.size()
		inputs_embeds = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(self.position_ids[:, 0:input_shape[1]])
		if token_type_ids is None:
			token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)
		embeddings = inputs_embeds + position_embeddings + token_type_embeddings
		embeddings = self.LayerNorm(embeddings)
		return embeddings

# assume the input shape is [batch_size, seq_len, hidden_size]
class BertPooler(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.dense = Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()
		self.pooling = BasePooler.from_pooling_type(PoolingType.CLS)

	def forward(
		self,
		hidden_states: Union[torch.Tensor, list[torch.Tensor]],
		prompt_lens: torch.Tensor = None
	) -> Union[torch.Tensor, list[torch.Tensor]]:
		pooled_output = self.pooling(hidden_states, prompt_lens)

		if isinstance(pooled_output, list):
			pooled_output = [self.activation(self.dense(x)) for x in pooled_output]
		else:
			pooled_output = self.activation(self.dense(pooled_output))
		return pooled_output
	
class BertIntermediate(nn.Module):

	def __init__(
		self,
		hidden_size: int,
		intermediate_size: int,	
	):
		super().__init__()
		self.dense = Linear(in_features=hidden_size, out_features=intermediate_size, bias=True)
		self.intermediate_act_fn = nn.GELU()

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		# Apply the dense layer and activation function
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states

class BertOutput(nn.Module):

	def __init__(
		self,
		hidden_size: int,
		intermediate_size: int,
		layer_norm_eps: float,
	):
		super().__init__()
		self.dense = Linear(in_features=intermediate_size, out_features=hidden_size, bias=True)
		self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

	def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
		# Apply the dense layer and activation function
		hidden_states = self.dense(hidden_states)
		
		# Add residual connection and apply layer normalization
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BertSelfOutput(nn.Module):

	def __init__(
		self,
		hidden_size: int,
		layer_norm_eps: float
	):
		super().__init__()
		self.dense = Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
		self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

	def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
		# Apply the dense layer and activation function
		hidden_states = self.dense(hidden_states)

		# Add residual connection and apply layer normalization
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states
	
class BertSelfAttention(nn.Module):
	_global_i = 0
	def __init__(
		self,
		hidden_size: int,
		num_attention_heads: int,
	):
		super().__init__()
		self.hidden_size = hidden_size
		self.total_num_heads = num_attention_heads
		self.num_heads = self.total_num_heads
		self.head_dim = self.hidden_size // self.total_num_heads
		self.total_num_kv_heads = self.total_num_heads
		self.num_kv_heads = max (1, self.total_num_kv_heads)

		self.q_size = self.num_heads * self.head_dim
		self.kv_size = self.num_kv_heads * self.head_dim
		self.scaling = self.head_dim ** -0.5
		self.i = 0
		self.qkv_proj = QKVLinear(
			hidden_size=self.hidden_size,
			q_size=self.q_size,
			kv_size=self.kv_size,
			bias=True
		)	
		self.attn = Attention(num_heads=self.num_heads,
								head_dim=self.head_dim,
								scale=self.scaling,
								num_kv_heads=self.num_kv_heads)

	def forward(
		self,
		hidden_states: torch.Tensor,
	) -> torch.Tensor:
		batch_size, seq_len, _ = hidden_states.size()
		qkv = self.qkv_proj(hidden_states)
		q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
		q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  
		v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
		# if BertSelfAttention._global_i == 0:
		# 	save_gpu_tensor_to_cpu(q, "/2023022031/Infer/pt/bert_custom_output", f"layer0_query_states")
		# 	save_gpu_tensor_to_cpu(k, "/2023022031/Infer/pt/bert_custom_output", f"layer0_key_states")
		# 	save_gpu_tensor_to_cpu(v, "/2023022031/Infer/pt/bert_custom_output", f"layer0_value_states")
		output = self.attn(q, k, v, True)
		output = output.transpose(1, 2)
		output = output.reshape(batch_size, seq_len, self.q_size)
		# if BertSelfAttention._global_i == 0:
		# 	save_gpu_tensor_to_cpu(output, "/2023022031/Infer/pt/bert_custom_output", f"bert_custom_output")
		# 	BertSelfAttention._global_i += 1
		return output
	
class BertAttention(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		num_attention_heads: int,
		layer_norm_eps: float,
	):
		super().__init__()
		self.self = BertSelfAttention(hidden_size, num_attention_heads)
		self.output = BertSelfOutput(hidden_size, layer_norm_eps)

	def forward(
		self,
		hidden_states: torch.Tensor,
		attention_mask: torch.Tensor = None,
	) -> torch.Tensor:
		self_output = self.self(hidden_states)
		return self.output(self_output, hidden_states)


class BertLayer(nn.Module):
	_global_i = 0
	def __init__(
		self,
		config: BertConfig,
	):
		super().__init__()
		self.attention = BertAttention(
			hidden_size=config.hidden_size,
			num_attention_heads=config.num_attention_heads,
			layer_norm_eps=config.layer_norm_eps,
		)

		self.intermediate = BertIntermediate(
			hidden_size=config.hidden_size,
			intermediate_size=config.intermediate_size,
		)

		self.output = BertOutput(
			hidden_size=config.hidden_size,
			intermediate_size=config.intermediate_size,
			layer_norm_eps=config.layer_norm_eps,
		)

	def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		# Apply attention
		attention_output = self.attention(hidden_states, attention_mask)
		# Apply intermediate and output layers
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		if BertLayer._global_i == 0:
			save_gpu_tensor_to_cpu(layer_output, "/2023022031/Infer/pt/bert_custom_output", f"layer0_output")
			BertLayer._global_i += 1

		return layer_output


class BertEncoder(nn.Module):
	def __init__(self, config: BertConfig):
		super().__init__()
		self.config = config
		self.num_layers = config.num_hidden_layers
		self.layer = nn.ModuleList(
			[BertLayer(config) for _ in range(self.num_layers)]
		)

	def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		i = 0
		for layer in self.layer:
			hidden_states = layer(hidden_states, attention_mask)
			if i == 21:
				return hidden_states
			i += 1
		return hidden_states

class BertModel(nn.Module):
	packed_modules_mapping = {"qkv_proj": ["query", "key", "value"]}
	def __init__(self, config: BertConfig):
		super().__init__()
		self.embeddings = BertEmbedding(
			config=config
		)
		self.encoder = BertEncoder(
			config=config
		)

	def forward(
		self, 
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor = None,
		position_ids: torch.Tensor = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		token_type_ids: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		if inputs_embeds is not None:
			hidden_states = inputs_embeds
		else:
			hidden_states = self.embeddings(
				input_ids=input_ids,
				position_ids=position_ids,
				token_type_ids=token_type_ids
			)
		# Pass through encoder
		encoder_output = self.encoder(hidden_states, attention_mask)
		
		return encoder_output
	