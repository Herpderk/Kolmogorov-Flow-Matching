from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import nn

from kolmogorov_flow_matching.utils import denormalize, normalize


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

    def __init__(
        self,
        backbone: nn.Module,
        mean: float = 0.0,
        std: float = 1.0,
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.normalize_flag = normalize_inputs

        # Register normalization stats as non-trainable persistent buffers
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return normalize(x, self.mean, self.std)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return denormalize(x, self.mean, self.std)

    def autoregressive_generation(
        self, init_conds: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """
        init_conds: shape [1, k_frames, H, W]
        """
        self.eval()
        history = init_conds.clone()
        predictions = []

        with torch.no_grad():
            for step in range(num_steps):
                # 1. Generate the next frame based on the current history window
                next_frame = self.sample(history)  # Shape: [1, 1, H, W]
                predictions.append(next_frame)

                # 2. Slide the window: Drop the oldest frame, append the new prediction
                # history[:, 1:] takes frames 1, 2, 3 (dropping 0)
                history = torch.cat([history[:, 1:], next_frame], dim=1)

        return torch.cat(predictions, dim=1)  # Shape: [1, num_steps, H, W]

    @abstractmethod
    def get_training_loss(
        self, x_target: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def sample(self, x_cond: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
