import torch
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
import triton
import triton.language as tl

@triton.jit
def store_kvcache_kernel(
	key_ptr,
	key_stride,
	value_ptr,
	value_stride,
	k_cache_ptr,
	v_cache_ptr,
	slot_mapping_ptr,
	D: tl.constexpr,
):
	idx = tl.program_id(0)
	key_offsets = idx * key_stride + tl.arange(0, D)
	value_offsets = idx * value_stride + tl.arange(0, D)
	key = tl.load(key_ptr + key_offsets)
	value = tl.load(value_ptr + value_offsets)
	slot = tl.load(slot_mapping_ptr + idx)
	cache_offsets = slot * D + tl.arange(0, D)
	tl.store(k_cache_ptr + cache_offsets, key)
	tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
	N, num_heads, head_dim = key.shape
	D = num_heads * head_dim
	assert key.stride(-1) == 1 and value.stride(-1) == 1
	assert key.stride(1) == head_dim and value.stride(1) == head_dim
	assert k_cache.stride(1) == D and v_cache.stride(1) == D
	assert slot_mapping.numel() == N
	store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def load_kvcache_kernel(
	k_cache_ptr,
	v_cache_ptr,
	output_k_ptr,
	output_v_ptr,
	slot_mapping_ptr,
	D: tl.constexpr,
):
	idx = tl.program_id(0)
	slot = tl.load(slot_mapping_ptr + idx)
	cache_offsets = slot * D + tl.arange(0, D)
	
	# 加载缓存中的KV值
	key = tl.load(k_cache_ptr + cache_offsets)
	value = tl.load(v_cache_ptr + cache_offsets)
	
	# 存储到输出
	output_offsets = idx * D + tl.arange(0, D)
	tl.store(output_k_ptr + output_offsets, key)
	tl.store(output_v_ptr + output_offsets, value)

def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, num_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
	N = slot_mapping.numel()
	D = num_heads * head_dim
	
	output_k = torch.empty((N, num_heads, head_dim), dtype=k_cache.dtype, device=k_cache.device)
	output_v = torch.empty((N, num_heads, head_dim), dtype=v_cache.dtype, device=v_cache.device)
	
	load_kvcache_kernel[(N,)](k_cache, v_cache, output_k, output_v, slot_mapping, D)
	return output_k, output_v

@dataclass
class KVCacheMetadata:
	"""存储KV Cache的元数据"""
	seq_id: int                  # 序列ID
	start_idx: int               # 在KV缓存中的起始索引
	length: int                  # 序列长度
	capacity: int                # 分配的总容量
	is_prefix: bool = False      # 是否是可共享前缀
	ref_count: int = 1           # 引用计数
	prefix_hash: Optional[str] = None  # 前缀哈希值
	
class FlashKVCacheManager:
	"""适用于Flash Attention的KV缓存管理器"""
	
	def __init__(
		self,
		num_layers: int,
		num_heads: int,
		head_dim: int,
		max_seq_len: int = 8192,
		max_batch_size: int = 8,
		enable_prefix_caching: bool = True,
		device: str = "cuda"
	):
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.head_dim = head_dim
		self.max_seq_len = max_seq_len
		self.max_batch_size = max_batch_size
		self.enable_prefix_caching = enable_prefix_caching
		self.device = device
		
		# 为每一层分配KV缓存
		self.k_cache = {}
		self.v_cache = {}
		for layer_idx in range(num_layers):
			# Flash Attention期望的格式: [batch_size*seq_len, num_heads, head_dim]
			self.k_cache[layer_idx] = torch.zeros(
				(max_batch_size * max_seq_len, num_heads, head_dim),
				dtype=torch.float16,
				device=device
			)
			self.v_cache[layer_idx] = torch.zeros(
				(max_batch_size * max_seq_len, num_heads, head_dim),
				dtype=torch.float16,
				device=device
			)
		
		# 序列元数据
		self.seq_metadata: Dict[int, KVCacheMetadata] = {}
		
		# 前缀缓存表
		self.prefix_hash_table: Dict[str, KVCacheMetadata] = {}
		
		# 空闲空间管理
		self.free_space: List[Tuple[int, int]] = [(0, max_batch_size * max_seq_len)]  # (start, length)
		
		# 统计信息
		self.cache_hits = 0
		self.cache_misses = 0
		
	def _allocate_space(self, length: int) -> Optional[int]:
		"""分配连续的缓存空间"""
		# 查找足够大的空闲空间
		for i, (start, free_len) in enumerate(self.free_space):
			if free_len >= length:
				# 找到合适的空间
				self.free_space[i] = (start + length, free_len - length)
				if self.free_space[i][1] == 0:
					self.free_space.pop(i)
				return start
		
		# 没有找到足够的空间
		return None
	
	def _free_space(self, start: int, length: int) -> None:
		"""释放空间并合并相邻的空闲块"""
		# 插入新的空闲块
		inserted = False
		for i, (free_start, free_len) in enumerate(self.free_space):
			if start + length == free_start:
				# 新块在现有块之前，合并
				self.free_space[i] = (start, free_len + length)
				inserted = True
				break
			elif free_start + free_len == start:
				# 新块在现有块之后，合并
				self.free_space[i] = (free_start, free_len + length)
				inserted = True
				break
		
		if not inserted:
			self.free_space.append((start, length))
		
		# 合并相邻的空闲块
		self.free_space.sort()
		i = 0
		while i < len(self.free_space) - 1:
			curr_start, curr_len = self.free_space[i]
			next_start, next_len = self.free_space[i+1]
			
			if curr_start + curr_len == next_start:
				# 合并块
				self.free_space[i] = (curr_start, curr_len + next_len)
				self.free_space.pop(i+1)
			else:
				i += 1
				
	def _compute_prefix_hash(self, token_ids: List[int], lora_id: int = 0) -> int:
		"""使用与vLLM一致的哈希计算方法"""
		hashed_tokens = tuple(token_ids)  # 转换为不可变类型
		return hash((hashed_tokens, lora_id))

	def allocate_for_sequence(
		self,
		seq_id: int,
		token_ids: List[int],
		extra_info: Optional[str] = None,
		min_capacity: Optional[int] = None
	) -> bool:
		"""为序列分配KV缓存空间"""
		if seq_id in self.seq_metadata:
			return True  # 已经分配

		length = len(token_ids)
		capacity = max(length * 2, min_capacity or length)  # 预留空间用于后续生成

		# 检查前缀缓存
		prefix_hash = None
		if self.enable_prefix_caching and token_ids:
			prefix_hash = self._compute_prefix_hash(token_ids, extra_info)
			if prefix_hash in self.prefix_hash_table:
				# 找到匹配的前缀
				prefix_meta = self.prefix_hash_table[prefix_hash]
				prefix_meta.ref_count += 1

				# 只需要为新token分配空间
				start_idx = self._allocate_space(capacity - length)
				if start_idx is None:
					return False  # 内存不足

				# 记录序列元数据
				self.seq_metadata[seq_id] = KVCacheMetadata(
					seq_id=seq_id,
					start_idx=start_idx,
					length=0,  # 当前没有新token
					capacity=capacity - length,
					is_prefix=False,
					prefix_hash=prefix_hash,
				)

				self.cache_hits += 1
				return True

		# 未找到匹配前缀或前缀缓存禁用，分配全新空间
		start_idx = self._allocate_space(capacity)
		if start_idx is None:
			return False  # 内存不足

		# 记录序列元数据
		self.seq_metadata[seq_id] = KVCacheMetadata(
			seq_id=seq_id,
			start_idx=start_idx,
			length=length,
			capacity=capacity,
			is_prefix=True if self.enable_prefix_caching else False,
			prefix_hash=prefix_hash,
		)

		# 如果启用了前缀缓存，将此序列记录为可共享前缀
		if self.enable_prefix_caching and prefix_hash:
			self.prefix_hash_table[prefix_hash] = self.seq_metadata[seq_id]

		self.cache_misses += 1
		return True

	def update_kv_cache(
		self,
		seq_id: int,
		layer_idx: int,
		k: torch.Tensor,  # [seq_len, num_heads, head_dim]
		v: torch.Tensor   # [seq_len, num_heads, head_dim]
	) -> None:
		"""更新指定序列和层的KV缓存"""
		if seq_id not in self.seq_metadata:
			raise ValueError(f"Sequence {seq_id} not found in cache")

		metadata = self.seq_metadata[seq_id]

		# 确定要写入的位置
		start_pos = metadata.start_idx + metadata.length
		seq_len = k.shape[0]

		if start_pos + seq_len - metadata.start_idx > metadata.capacity:
			raise ValueError(f"KV cache overflow for sequence {seq_id}")

		# 创建slot映射
		slot_mapping = torch.arange(start_pos, start_pos + seq_len, device=k.device, dtype=torch.int32)
		store_kvcache(k, v, self.k_cache[layer_idx], self.v_cache[layer_idx], slot_mapping)
		# 更新序列长度
		metadata.length += seq_len

	def get_kv_cache_for_attention(
		self, 
		seq_id: int,
		layer_idx: int
	) -> Tuple[torch.Tensor, torch.Tensor, int]:
		"""使用Triton kernel获取指定序列和层的KV缓存"""
		if seq_id not in self.seq_metadata:
			raise ValueError(f"Sequence {seq_id} not found in cache")
		
		metadata = self.seq_metadata[seq_id]
		
		# 检查是否有前缀
		prefix_len = 0
		slots = []
		
		# 获取前缀的slots
		if metadata.prefix_hash and metadata.prefix_hash in self.prefix_hash_table:
			prefix_meta = self.prefix_hash_table[metadata.prefix_hash]
			prefix_len = prefix_meta.length
			prefix_slots = torch.arange(prefix_meta.start_idx, 
									   prefix_meta.start_idx + prefix_len,
									   device=self.device, dtype=torch.int32)
			slots.append(prefix_slots)
		
		# 获取序列自己的slots
		if metadata.length > 0:
			self_slots = torch.arange(metadata.start_idx, 
									 metadata.start_idx + metadata.length,
									 device=self.device, dtype=torch.int32)
			slots.append(self_slots)
		
		# 合并所有slots
		all_slots = torch.cat(slots) if slots else torch.tensor([], device=self.device, dtype=torch.int32)
		
		# 使用Triton kernel读取KV值
		k, v = load_kvcache(self.k_cache[layer_idx], self.v_cache[layer_idx], 
						   all_slots, self.num_heads, self.head_dim)
		
		return k, v, prefix_len