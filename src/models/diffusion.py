from typing import Tuple

import torch
from torch import nn


class Diffusion(nn.Module):
    def __init__(
        self,
        T: int = 1000,  # total number of diffusion steps,
        b_0: float = 1e-4,
        b_T: float = 2e-2,
        n_data_dims: int = 3,  # number of data dimensions. For example, colored image data has 3 (channel, height, width)
    ):
        super().__init__()
        self.T = T

        # calculate the 1D tensor for beta containing the values for each diffusion step
        # using quadratic schedule
        beta = torch.linspace(b_0**0.5, b_T**0.5, T) ** 2

        # based on n_data_dims, make the shape of beta broadcastable to batched data
        for _ in range(n_data_dims):
            beta = beta.unsqueeze(-1)

        # calculate alpha and alpha_bar from beta
        # both alpha and alpha_bar have T elements as well, one for each diffusion step
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim=0)

        # register the tensors as buffers to be saved with the model
        # and to be moved to the right device when calling .to(device)
        # You can access them like a normal attribute, like self.alpha
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("beta", beta)

    @torch.no_grad()
    def forward(
        self,
        x_0: torch.FloatTensor,  # (batch_size, *data_shape),
        t: torch.LongTensor,  # (batch_size,),
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor
    ]:  # noisy data and the epsilon used to corrupt it
        """
        for each data sample in the batch, draw a sample from q(x_t|x_0, t)
        according to the schedule and the corresponding diffusion step of each data sample.

        You can index alpha, alpha_bar, or beta with the tensor t directly,
        and get a batch of alpha, alpha_bar, or beta values.

        Returns:
        x_t: torch.FloatTensor, the corrupted batch
        eps_q: torch.FloatTensor, the noise used to corrupt the data
        """

        # mean of q(x_t|x_0, t)
        mu = x_0 * self.alpha_bar[t].sqrt()

        # std of q(x_t|x_0, t)
        std = (1 - self.alpha_bar[t]).sqrt()

        # sample from q using the reparameterization trick
        eps_q = torch.randn_like(x_0)
        x_t = mu + std * eps_q

        return x_t, eps_q

    @torch.inference_mode()
    def reverse(
        self,
        x_t: torch.FloatTensor,  # (batch_size, *data_shape),
        t: int,
        eps_theta: torch.FloatTensor,  # (batch_size, *data_shape),
    ):
        """
        for a batch of corrupted data x_t and using the estimated noise eps_theta,
        sample from p(x_{t-1}|x_t, t)

        Here, t is the same for all samples in the batch.

        Returns:
        x_t_1: torch.FloatTensor, a single-step denoised batch of data
        """

        # mean of p(x_{t-1}|x_t, t)
        mu = (
            x_t - eps_theta * self.beta[t] / (1 - self.alpha_bar[t]).sqrt()
        ) / self.alpha[t].sqrt()

        # std of p(x_{t-1}|x_t, t)
        std = self.beta[t].sqrt()

        # sample from p using the reparameterization trick
        # NOTE: no noise is added at the final denoising step (t=0 -> eps_p=0)
        eps_p = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t_1 = mu + std * eps_p

        return x_t_1
