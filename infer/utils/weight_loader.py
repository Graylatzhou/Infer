import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def load_weights_from_safetensors(model: nn.Module, path: str) -> None:
    