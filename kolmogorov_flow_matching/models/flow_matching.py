import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from kolmogorov_flow_matching.models.base import (
    ConditionalBackbone,
    ConditionalGenerativeFramework,
)


class ConditionalFlowMatching(ConditionalGenerativeFramework):
    def __init__(
        self,
        backbone: ConditionalBackbone,
        mean: float = 0.0,
        std: float = 1.0,
        normalize_inputs: bool = True,
    ):
        super().__init__(backbone, mean, std, normalize_inputs)

    def get_training_loss(
        self, x_target: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        x_target = self.normalize(x_target) if self.normalize_flag else x_target
        x_cond = self.normalize(x_cond) if self.normalize_flag else x_cond

        B = x_target.size(0)
        device = x_target.device

        z_0 = torch.randn_like(x_target)
        s = torch.rand((B,), device=device)
        s_view = s
        while s_view.ndim < x_target.ndim:
            s_view = s_view.unsqueeze(-1)

        z_s = (1.0 - s_view) * z_0 + s_view * x_target
        target_velocity = x_target - z_0

        # Inject small noise into the condition to correct drifting states
        if self.training:
            noise_scale = 0.02
            x_cond_input = x_cond + torch.randn_like(x_cond) * noise_scale
        else:
            x_cond_input = x_cond

        predicted_velocity = self.backbone(s, z_s, x_cond_input)
        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.inference_mode()
    def sample(
        self,
        x_cond: torch.Tensor,
        method: str = "euler",
        integration_steps: int = 10,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        x_cond = self.normalize(x_cond) if self.normalize_flag else x_cond

        B = x_cond.size(0)
        device = x_cond.device
        z_0 = torch.randn((B, *self.backbone.data_shape), device=device)

        self.backbone.eval()

        def ode_func(s_scalar: torch.Tensor, z_current: torch.Tensor) -> torch.Tensor:
            s_tensor = torch.full((B,), s_scalar.item(), device=device)
            return self.backbone(s_tensor, z_current, x_cond)

        s_grid = torch.linspace(0.0, 1.0, integration_steps + 1, device=device)

        trajectory = odeint(
            func=ode_func,
            y0=z_0,
            t=s_grid,
            method=method,
            options={"step_size": 1.0 / integration_steps}
            if method in ["euler", "rk4"]
            else {},
        )

        self.backbone.train()

        output = trajectory if return_trajectory else trajectory[-1]
        return self.denormalize(output) if self.normalize_flag else output
