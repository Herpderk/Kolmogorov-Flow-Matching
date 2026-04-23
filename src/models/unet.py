from typing import Sequence, Union

import torch
from torch import nn
from torch.nn import functional as F

from src.models.diffusion import Diffusion


class ConvBlock(nn.Module):
    """
    a module in the contracting path of the U-Net.
    simply consists of a series of conv, batchnorm, and activation layers.
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
                    padding="same",
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


class DiffusionUnet(nn.Module):
    """
    A U-Net with a corresponding diffusion step decoder for each block.
    """

    def __init__(
        self,
        data_shape: Sequence[int] = [1, 32, 32],
        # diffusion parameters
        T: int = 1000,
        b_0: float = 1e-4,
        b_T: float = 2e-2,
        # diffusion step embedding
        t_embed_dim: int = 128,
        # U-Net architecture
        channels: Sequence[int] = [16, 32, 64, 128],
        n_block_layers: int = 2,
        activation_name: str = "ReLU",
        batchnorm: bool = False,
    ):
        super().__init__()

        # to be used for data generation
        self.data_shape = data_shape

        # ======================= Diffusion ==========================

        self.diffusion = Diffusion(
            T=T,
            b_0=b_0,
            b_T=b_T,
            n_data_dims=len(data_shape),
        )

        # =========================== Model ===========================

        n_layers = len(channels)
        self.n_layers = n_layers

        self.blocks = nn.ModuleDict()

        self.t_emebdder = nn.Embedding(
            num_embeddings=T,
            embedding_dim=t_embed_dim,
        )

        self.t_decoder = nn.ModuleDict()
        # An encoder should be assigned to the output of each block in
        # the contracting path or the expanding path.
        # Each encoder takes as input the embedded diffusion step.

        # contracting path
        for i in range(n_layers):
            # example for 4 layers:
            # down_0, down_1, down_2, down_3
            self.blocks[f"down_{i}"] = ConvBlock(
                in_channels=data_shape[0] if i == 0 else channels[i - 1],
                out_channels=channels[i],
                activation_name=activation_name,
                n_layers=n_block_layers,
                batchnorm=batchnorm,
            )

            # the output of the t_encoder will be added to the output of the block
            self.t_decoder[f"down_{i}"] = FeedForward(
                in_features=t_embed_dim,
                out_features=channels[i],
                hidden_sizes=[channels[i] // 2],
                activation_name=activation_name,
            )

        # expanding path (reverse depth)
        for i in range(n_layers - 2, -1, -1):
            # example for 4 layers:
            # up_2, up_1, up_0

            self.blocks[f"up_{i}"] = UpBlock(
                in_channels=channels[i + 1],
                skip_channels=channels[i],
                out_channels=channels[i],
                activation_name=activation_name,
                n_layers=n_block_layers,
                batchnorm=batchnorm,
            )

            # the output of the t_encoder will be added to the output of the block
            self.t_decoder[f"up_{i}"] = FeedForward(
                in_features=t_embed_dim,
                out_features=channels[i],
                hidden_sizes=[channels[i] // 2],
                activation_name=activation_name,
            )

        # final output layer to get the estimated noise tensor
        # the output should have the same shape as the input data
        # use kernel_size = 1, stride = 1, padding = 0
        self.out = nn.Conv2d(
            in_channels=channels[0],
            out_channels=data_shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.FloatTensor,  # (batch_size, *data_shape)
        t: Union[int, torch.LongTensor],  # (batch_size,)
    ) -> torch.FloatTensor:  # (batch_size, *data_shape)
        """
        Inputs:
            x: a batch of corrputed data
            t: the corresponding diffusion step for each sample in the batch

        returns:
            eps_theta: the estimated noise tensor used to corrupt the data in the forward diffusion.
        """

        if isinstance(t, int):
            # create a LongTensor of shape (batch_size,) from t, on the same device as x_0
            t = torch.tensor(len(x) * [t], device=x.device, dtype=torch.long)

        # embed the diffusion step
        t_embedded = self.t_emebdder(t)

        # to store the skip connections
        skips = []

        # contracting path
        for i in range(self.n_layers):
            # pass data through the block
            x = self.blocks[f"down_{i}"](x)

            # append the result to skips, to be used in the expanding path
            # except for the last down block (deepest layer)
            if i < self.n_layers - 1:
                skips.append(x)

            # encode t_embedded with the corresponding decoder
            # and add it channel-wise to data
            x = x + self.t_decoder[f"down_{i}"](t_embedded)

            # downsample data with F.avg_pool2d and kernel_size=2
            # except for the last down block (deepest layer)
            if i < self.n_layers - 1:
                x = F.avg_pool2d(x, 2)

        # expanding path (reverse depth)
        for i in range(self.n_layers - 2, -1, -1):
            # pass the data and the corresponding skip connection to the block
            x = self.blocks[f"up_{i}"](x, skips.pop())

            # encode t_embedded with the corresponding decoder
            # and add it channel-wise to the data
            x = x + self.t_decoder[f"up_{i}"](t_embedded)

        # pass data through the final convolutional layer
        eps_theta = self.out(x)

        return eps_theta

    @torch.inference_mode()
    def generate(
        self,
        n_samples: int,
        device: str,
    ) -> torch.FloatTensor:  # (n_samples, *data_shape):
        """
        Sample a noise tensor of the right shape and device.
        Execute the reverse diffusion process using the model.

        Returns:
           xs of shape (T, n_samples, *data_shape)
           the full batch of denoised samples at each diffusion step
           starting from pure noise and ending with the final generated samples.
        """
        self.eval().to(device)

        # start from pure noise of the right shape and device
        x = torch.randn(n_samples, *self.data_shape, device=device)

        # for the sake of visualization, we will store the data throughout reverse diffusion
        xs = [x]

        # denoise step-by-step by sampling from p(x_{t-1}|x_t, t)
        # t = T-1, T-2, ..., 1, 0
        for t in range(self.diffusion.T - 1, -1, -1):
            # get the model output
            eps_theta = self(x, t)

            # do one step of reverse diffusion
            x = self.diffusion.reverse(x, t, eps_theta)

            # append the denoised data to xs
            xs.append(x)

        return torch.stack(xs)
