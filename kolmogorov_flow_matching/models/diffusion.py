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
        super().__init__(backbone, mean, std, normalize_inputs)

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
        num_inference_steps: int = 50,
        eta: float = 0.0,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        x_cond = self.normalize(x_cond) if self.normalize_flag else x_cond

        B = x_cond.size(0)
        device = x_cond.device
        x = torch.randn((B, *self.backbone.data_shape), device=device)
        trajectory = [x] if return_trajectory else None

        # Create the sub-sampled DDIM time steps
        step_ratio = self.T // num_inference_steps
        timesteps = (
            torch.flip(
                (torch.arange(0, num_inference_steps) * step_ratio).round(), dims=[0]
            )
            .to(device)
            .long()
        )
        timesteps_prev = torch.cat([timesteps[1:], torch.tensor([-1], device=device)])

        # Iterate over the shortened sequence
        for t_step, t_prev in zip(timesteps, timesteps_prev):
            t_tensor = torch.full(
                (B,), t_step.item(), device=device, dtype=torch.float32
            )
            predicted_noise = self.backbone(t_tensor, x, x_cond)

            # Grab the alpha_bar for current and previous step
            alpha_bar_t = self.alpha_bar[t_step]
            alpha_bar_prev = (
                self.alpha_bar[t_prev]
                if t_prev >= 0
                else torch.tensor(1.0, device=device)
            )

            # DDIM Equation
            # Predict the clean image
            pred_x0 = (
                x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)

            # Calculate standard deviation of noise (sigma_t)
            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )

            # Calculate the direction pointing to x_t
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma_t**2) * predicted_noise

            # Jump to the previous timestep
            noise = torch.randn_like(x) if t_prev >= 0 else 0.0
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

            if return_trajectory:
                trajectory.append(x)

        output = torch.stack(trajectory) if return_trajectory else x
        return self.denormalize(output) if self.normalize_flag else output
