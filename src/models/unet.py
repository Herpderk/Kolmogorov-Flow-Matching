import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.backbone import ConditionalBackbone


class ConvBlock(nn.Module):
    """
    Modified ConvBlock with circular padding for periodic Kolmogorov flow boundaries.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_name: str = "ReLU",
        n_layers: int = 2,
        batchnorm: bool = False,
    ):
        super().__init__()
        activation = nn.__getattribute__(activation_name)
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,  # explicit padding of 1
                    padding_mode="circular",  # CRITICAL: Periodic boundary conditions
                )
            )
            if batchnorm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(activation())

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> torch.FloatTensor:

        return self.layers(x)


class UpBlock(nn.Module):
    """
    a module in the expanding path of the U-Net.
    First, the input to the block is upconved and concatenated with the skip connection.
    Then, a series of conv, batchnorm, and activation layers are applied.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        activation_name: str = "ReLU",
        n_layers: int = 2,
        batchnorm: bool = False,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.layers = ConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            activation_name=activation_name,
            n_layers=n_layers,
            batchnorm=batchnorm,
        )

    def forward(
        self,
        x: torch.FloatTensor,
        skip: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # upconv x
        x = self.up(x)

        # concatenate the output with the skip connection
        x = torch.cat([skip, x], dim=1)

        # pass x through the main block and return the result
        return self.layers(x)


class FeedForward(nn.Module):
    """
    A simple feedforward neural network to decode the embedded diffusion step in each stage.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: Sequence[int],
        activation_name: str = "ReLU",
    ):
        super().__init__()
        activation = nn.__getattribute__(activation_name)
        n_layers = len(hidden_sizes)
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.append(
                nn.Linear(
                    in_features=in_features if i == 0 else hidden_sizes[i - 1],
                    out_features=hidden_sizes[i],
                )
            )
            self.layers.append(activation())

        self.layers.append(
            nn.Linear(
                in_features=hidden_sizes[-1],
                out_features=out_features,
            )
        )

    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, in_features)
    ) -> torch.FloatTensor:  # (batch_size, out_features, 1, 1)
        """
        The output is going to be added to data of shape (B, C, H, W)
        where C = out_features,
        So it has to be broadcastable to the same shape.
        """
        return self.layers(x)[..., None, None]


class SinusoidalPositionEmbedding(nn.Module):
    """
    Accepts continuous or discrete time inputs and maps them to a high-dimensional space.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time can be shape (batch_size,) containing floats like 0.45 or ints like 500
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUnetBackbone(ConditionalBackbone):
    """
    A purely functional U-Net that inherits from ConditionalBackbone.
    """

    def __init__(
        self,
        data_shape: Sequence[int] = [1, 32, 32],
        k_frames: int = 4,  # History condition window
        # Embedding dimension for the time variable
        t_embed_dim: int = 128,
        # U-Net architecture
        channels: Sequence[int] = [16, 32, 64, 128],
        n_block_layers: int = 2,
        activation_name: str = "ReLU",
        batchnorm: bool = False,
    ):

        # 1. Initialize the Base Class (handles self.data_shape and nn.Module setup)
        super().__init__(data_shape=data_shape)

        # ================== U-Net Setup ==================
        n_layers = len(channels)
        self.n_layers = n_layers
        self.blocks = nn.ModuleDict()

        # Continuous time embedder
        self.t_embedder = nn.Sequential(
            SinusoidalPositionEmbedding(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.__getattribute__(activation_name)(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        self.t_decoder = nn.ModuleDict()

        # contracting path
        for i in range(n_layers):
            in_ch = (data_shape[0] * (k_frames + 1)) if i == 0 else channels[i - 1]

            self.blocks[f"down_{i}"] = ConvBlock(
                in_channels=in_ch,
                out_channels=channels[i],
                activation_name=activation_name,
                n_layers=n_block_layers,
                batchnorm=batchnorm,
            )

            self.t_decoder[f"down_{i}"] = FeedForward(
                in_features=t_embed_dim,
                out_features=channels[i],
                hidden_sizes=[channels[i] // 2],
                activation_name=activation_name,
            )

        # expanding path
        for i in range(n_layers - 2, -1, -1):
            self.blocks[f"up_{i}"] = UpBlock(
                in_channels=channels[i + 1],
                skip_channels=channels[i],
                out_channels=channels[i],
                activation_name=activation_name,
                n_layers=n_block_layers,
                batchnorm=batchnorm,
            )

            self.t_decoder[f"up_{i}"] = FeedForward(
                in_features=t_embed_dim,
                out_features=channels[i],
                hidden_sizes=[channels[i] // 2],
                activation_name=activation_name,
            )

        # final output layer
        self.out = nn.Conv2d(
            in_channels=channels[0],
            out_channels=data_shape[
                0
            ],  # Must output exactly the target channel dimension
            kernel_size=1,
            stride=1,
            padding=0,
        )

    # 2. Implement the required abstract method
    def forward(
        self, time: torch.FloatTensor, x: torch.FloatTensor, x_cond: torch.FloatTensor
    ) -> torch.FloatTensor:

        t_embedded = self.t_embedder(time)

        # Condition on history via channel concatenation
        x = torch.cat([x, x_cond], dim=1)

        skips = []

        # Contracting Path
        for i in range(self.n_layers):
            x = self.blocks[f"down_{i}"](x)
            if i < self.n_layers - 1:
                skips.append(x)
            x = x + self.t_decoder[f"down_{i}"](t_embedded)
            if i < self.n_layers - 1:
                x = F.avg_pool2d(x, 2)

        # Expanding Path
        for i in range(self.n_layers - 2, -1, -1):
            x = self.blocks[f"up_{i}"](x, skips.pop())
            x = x + self.t_decoder[f"up_{i}"](t_embedded)

        return self.out(x)
