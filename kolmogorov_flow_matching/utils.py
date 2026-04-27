import torch


def normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


def denormalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x * std) + mean
