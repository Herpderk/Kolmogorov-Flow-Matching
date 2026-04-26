from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import nn


class ConditionalBackbone(nn.Module, ABC):
    """
    Abstract base class for all generative backbones (U-Nets, Transformers, etc.).
    Enforces the required attributes and method signatures expected by
    the Diffusion and Flow Matching wrapper classes.
    """

    def __init__(self, data_shape: Sequence[int]):
        super().__init__()
        # Required by the Framework's sample() method to generate initial noise
        self.data_shape = data_shape

    @abstractmethod
    def forward(
        self,
        time: torch.Tensor,
        x: torch.Tensor,
        x_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Must be implemented by subclasses.

        Args:
            x: (B, C, H, W) The active state (noisy image or interpolated state).
            time: (B,) The continuous float s in [0,1] or discrete integer t.
            x_cond: (B, k*C, H, W) The historical condition frames.

        Returns:
            Tensor of shape (B, C, H, W) representing noise (eps) or velocity (v).
        """
        pass


class ConditionalGenerativeFramework(nn.Module, ABC):
    """
    Abstract base class that handles neural network storage and
    persistent physical data normalization statistics.
    """

    def __init__(self, backbone: nn.Module, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.backbone = backbone

        # Register normalization stats as non-trainable persistent buffers
        self.register_buffer("data_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("data_std", torch.tensor(std, dtype=torch.float32))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Scales physical fluid data to N(0, 1) latent space."""
        return (x - self.data_mean) / self.data_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Unscales N(0, 1) latent data back to physical fluid metrics."""
        return (x * self.data_std) + self.data_mean

    @abstractmethod
    def get_training_loss(
        self, x_target: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, x_cond: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
