from typing import Any
from numpy import einsum
import torch

class FuncFactory:
    def __init__(self, noise_percent = 0):
        self.noise_percent = noise_percent
    
    def matmul_with_noise(self, first: torch.Tensor, second: torch.Tensor):
        result = torch.matmul(first, second)
    
        if self.noise_percent == 0:
            return result
    
        result_with_noise = result + torch.randn_like(result) * (result.max() * self.noise_percent)
        return result_with_noise

    def einsum_with_noise(self, desc, first: torch.Tensor | Any, second: torch.Tensor | Any):
        result = torch.einsum(desc, (first, second))
    
        if self.noise_percent == 0:
            return result

        result_with_noise = result + torch.randn_like(result) * (result.max() * self.noise_percent)
        return result_with_noise
        




