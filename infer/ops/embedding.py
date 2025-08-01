import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from infer.context import get_context

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = F.embedding(input, self.weight)
        return y

class LMHead(Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
	):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            input = input[last_indices].contiguous()
        logits = F.linear(input, self.weight)
        return logits
