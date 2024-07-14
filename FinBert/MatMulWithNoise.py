from typing import Any
import torch

@staticmethod
def matmul_with_noise(one: torch.Tensor, other: torch.Tensor, noise_percent = 0):
    result = torch.matmul(one, other)
    
    if noise_percent == 0:
        return result
    
    result_with_noise = result + torch.randn_like(result) * (result.max() * noise_percent)
    return result_with_noise

@staticmethod
def einsum_with_noise(desc, one: torch.Tensor | Any, two: torch.Tensor | Any, noise_percent = 0):
    result = torch.einsum(desc, (one, two))
    
    if noise_percent == 0:
        return result

    result_with_noise = result + torch.randn_like(result) * (result.max() * noise_percent)
    return result_with_noise

