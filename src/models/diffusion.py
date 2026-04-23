import torch
from torch import nn

from src.models.backbone import ConditionalBackbone


class ConditionalDiffusion(nn.Module):
    def __init__(
        self,
        backbone: ConditionalBackbone,
        T: int = 1000,
        b_0: float = 1e-4,
        b_T: float = 2e-2,
    ):
        super().__init__()
        self.backbone = backbone
        self.T = T

        # Precompute diffusion schedules
        beta = torch.linspace(b_0**0.5, b_T**0.5, T) ** 2
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register_buffer("sqrt_recip_alpha", (1.0 / alpha).sqrt())

    def get_training_loss(
        self, x_target: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the MSE loss for the noise prediction.
        x_target: (B, C, H, W) The true next frame u_{t+1}
        x_cond: (B, k*C, H, W) The historical frames [u_{t-k+1}, ..., u_t]
        """
        B = x_target.size(0)
        device = x_target.device

        # 1. Sample random timesteps for each item in the batch
        t = torch.randint(0, self.T, (B,), device=device).long()

        # 2. Sample random Gaussian noise
        noise = torch.randn_like(x_target)

        # 3. Add noise to the target according to the schedule
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(B, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)

        x_noisy = sqrt_alpha_bar_t * x_target + sqrt_one_minus_alpha_bar_t * noise

        # 4. Predict the noise using the backbone (pass float(t) to use sinusoidal embeddings nicely)
        predicted_noise = self.backbone(t.float(), x_noisy, x_cond)

        # 5. Return MSE Loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        return loss

    @torch.inference_mode()
    def sample(
        self, x_cond: torch.Tensor, return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Autoregressively predicts the next frame given the condition.
        """
        B = x_cond.size(0)
        device = x_cond.device

        # Start from pure noise
        x = torch.randn((B, *self.backbone.data_shape), device=device)
        trajectory = [x] if return_trajectory else None

        self.backbone.eval()

        # Reverse diffusion loop
        for t_step in reversed(range(self.T)):
            t_tensor = torch.full((B,), t_step, device=device, dtype=torch.float32)

            # Predict noise
            predicted_noise = self.backbone(t_tensor, x, x_cond)

            # DDPM reverse step
            beta_t = self.beta[t_step]
            alpha_t = self.alpha[t_step]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_step]
            sqrt_recip_alpha_t = self.sqrt_recip_alpha[t_step]

            # Mean calculation
            model_mean = sqrt_recip_alpha_t * (
                x - beta_t / sqrt_one_minus_alpha_bar_t * predicted_noise
            )

            if t_step > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(beta_t) * noise
            else:
                x = model_mean

            if return_trajectory:
                trajectory.append(x)

        self.backbone.train()
        return torch.stack(trajectory) if return_trajectory else x
