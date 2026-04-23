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
