from typing import Any
import torch

def matmul_with_noise(input: torch.Tensor, other: torch.Tensor):
    return torch.matmul(input, other)

def einsum_with_noise(*args: Any):
    return 


