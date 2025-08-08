import torch.nn as nn
from enum import IntEnum
from abc import ABC, abstractmethod # for constructing abstract base classes
import torch
from typing import Union, Optional

class PoolingType(IntEnum):
	CLS = 0


class BasePooler(nn.Module, ABC):

	@staticmethod
	def from_pooling_type(pooling_type: str) -> "BasePooler":
		if pooling_type == "cls":
			return CLSPool()
		else:
			raise ValueError(f"Unsupported pooling type: {pooling_type}")

	@abstractmethod
	def forward_one(
		self,
		hidden_states: torch.Tensor,
		prompt_len: Optional[torch.Tensor] = None
	) -> torch.Tensor:
		"""处理单个序列的池化操作"""
		raise NotImplementedError
	
	@abstractmethod
	def forward_all(
		self,
		hidden_states: torch.Tensor,
		prompt_lens: torch.Tensor
	) -> Union[list[torch.Tensor], torch.Tensor]:
		"""处理多个序列的池化操作"""
		raise NotImplementedError

	def forward(
		self,
		hidden_states: Union[torch.Tensor, list[torch.Tensor]],
		prompt_lens: torch.Tensor
	) -> Union[list[torch.Tensor], torch.Tensor]:
		if isinstance(hidden_states, list):
			return [
				self.forward_one(h, prompt_len)
				for h, prompt_len in zip(hidden_states, prompt_lens)
			]
		return self.forward_all(hidden_states, prompt_lens)


class CLSPool(BasePooler):

	def forward_one(
		self,
		hidden_states: torch.Tensor,
		prompt_len: Optional[torch.Tensor] = None
	) -> torch.Tensor:
		return hidden_states[-1]

	# varlen
	def forward_all(
		self,
		hidden_states: torch.Tensor,
		prompt_lens: torch.Tensor
	) -> Union[list[torch.Tensor], torch.Tensor]:
		first_token_flat_indices = torch.zero_like(prompt_lens)
		first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
		return hidden_states[first_token_flat_indices]
