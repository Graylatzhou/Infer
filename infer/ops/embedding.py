from CustomOp import CustomOp
import torch
from infer_ops import embedding
from typing import Optional

class Embedding(CustomOp):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader():
        pass

    def forward_native(self, input: torch.Tensor) -> torch.Tensor:
        y = torch.empty(input.shape[0], self.embedding_dim, device=input.device, dtype=self.weight.dtype)
        embedding.embedding(input, self.weight, y)
        return y
    
    def forward_cuda(self, input: torch.Tensor) -> torch.Tensor:
        y = torch.empty(input.shape[0], self.embedding_dim, device=input.device, dtype=self.weight.dtype)
        embedding.embedding_cuda(input, self.weight, y)
        return y