import torch
import torch.nn.functional as F

from kolmogorov_flow_matching.models.base import (
    ConditionalBackbone,
    ConditionalGenerativeFramework,
)


class ConditionalDiffusion(ConditionalGenerativeFramework):
    def __init__(
        self,
        backbone: ConditionalBackbone,
        mean: float = 0.0,
        std: float = 1.0,
        T: int = 1000,
        b_0: float = 1e-4,
        b_T: float = 2e-2,
        normalize_inputs: bool = True,
    ):
        # Passes the backbone and stats to ConditionalGenerativeFramework
        super().__init__(backbone, mean, std)

        self.normalize_flag = normalize_inputs
        self.T = T
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
        x_target = self.normalize(x_target) if self.normalize_flag else x_target
        x_cond = self.normalize(x_cond) if self.normalize_flag else x_cond

        B = x_target.size(0)
        device = x_target.device

        t = torch.randint(0, self.T, (B,), device=device).long()
        noise = torch.randn_like(x_target)

        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(B, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(B, 1, 1, 1)
        x_noisy = sqrt_alpha_bar_t * x_target + sqrt_one_minus_alpha_bar_t * noise

        predicted_noise = self.backbone(t.float(), x_noisy, x_cond)
        return F.mse_loss(predicted_noise, noise)

    @torch.inference_mode()
    def sample(
        self,
        x_cond: torch.Tensor,
        return_physical: bool = True,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        x_cond = self.normalize(x_cond) if self.normalize_flag else x_cond

        B = x_cond.size(0)
        device = x_cond.device
        x = torch.randn((B, *self.backbone.data_shape), device=device)
        trajectory = [x] if return_trajectory else None

        self.backbone.eval()

        for t_step in reversed(range(self.T)):
            t_tensor = torch.full((B,), t_step, device=device, dtype=torch.float32)
            predicted_noise = self.backbone(t_tensor, x, x_cond)

            beta_t = self.beta[t_step]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t_step]
            sqrt_recip_alpha_t = self.sqrt_recip_alpha[t_step]

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

        output = torch.stack(trajectory) if return_trajectory else x

        if return_physical and self.normalize_flag:
            output = self.denormalize(output)

        return output
