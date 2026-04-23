import torch
from torch import nn
from torchdiffeq import odeint

from src.models.backbone import ConditionalBackbone


class ConditionalFlowMatching(nn.Module):
    def __init__(self, backbone: ConditionalBackbone, integration_steps: int = 10):
        super().__init__()
        self.backbone = backbone
        self.integration_steps = integration_steps

    def get_training_loss(
        self, x_target: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the MSE loss for the vector field prediction.
        x_target: (B, C, H, W) The true next frame u_{t+1} (z_1)
        x_cond: (B, k*C, H, W) The historical frames [u_{t-k+1}, ..., u_t]
        """
        B = x_target.size(0)
        device = x_target.device

        # 1. Sample pure noise (z_0)
        z_0 = torch.randn_like(x_target)

        # 2. Sample continuous integration times s in [0, 1]
        s = torch.rand((B,), device=device)
        s_view = s.view(B, 1, 1, 1)

        # 3. Interpolate between noise and target (The Optimal Transport path)
        # z_s = (1 - s) * z_0 + s * z_1
        z_s = (1.0 - s_view) * z_0 + s_view * x_target

        # 4. Calculate the true target velocity (derivative of the path)
        target_velocity = x_target - z_0

        # 5. Predict the velocity using the backbone
        predicted_velocity = self.backbone(s, z_s, x_cond)

        # 6. Return MSE Loss
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    @torch.inference_mode()
    def sample(
        self, x_cond: torch.Tensor, return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Predicts the next frame by solving the ODE from s=0 to s=1 using the Euler method.
        """
        B = x_cond.size(0)
        device = x_cond.device

        # Start from pure noise at s = 0
        z = torch.randn((B, *self.backbone.data_shape), device=device)
        trajectory = [z] if return_trajectory else None

        self.backbone.eval()

        # Simple Euler ODE Integration
        ds = 1.0 / self.integration_steps

        for step in range(self.integration_steps):
            # Current time s
            s_val = step * ds
            s_tensor = torch.full((B,), s_val, device=device, dtype=torch.float32)

            # Predict velocity vector field v(z, s, condition)
            velocity = self.backbone(s_tensor, z, x_cond)

            # Euler step: z_{s+ds} = z_s + v * ds
            z = z + velocity * ds

            if return_trajectory:
                trajectory.append(z)

        self.backbone.train()
        return torch.stack(trajectory) if return_trajectory else z


class ConditionalFlowMatching(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # We remove integration_steps from __init__ because adaptive
        # solvers determine their own step counts dynamically.

    def get_training_loss(
        self, x_target: torch.Tensor, x_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the MSE loss for the vector field prediction.
        """
        B = x_target.size(0)
        device = x_target.device

        # 1. Sample pure noise (z_0)
        z_0 = torch.randn_like(x_target)

        # 2. Sample continuous integration times s in [0, 1]
        s = torch.rand((B,), device=device)
        s_view = s.view(B, 1, 1, 1)

        # 3. Interpolate between noise and target (The Optimal Transport path)
        z_s = (1.0 - s_view) * z_0 + s_view * x_target

        # 4. Calculate the true target velocity
        target_velocity = x_target - z_0

        # 5. Predict the velocity using the backbone
        # (Assuming signature: time, state, condition)
        predicted_velocity = self.backbone(s, z_s, x_cond)

        # 6. Return MSE Loss
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    @torch.inference_mode()
    def sample(
        self,
        x_cond: torch.Tensor,
        method: str = "euler",
        integration_steps: int = 10,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Predicts the next frame by solving the ODE from s=0 to s=1.
        Now completely agnostic to the solver method.
        """
        B = x_cond.size(0)
        device = x_cond.device

        # Start from pure noise at s = 0
        z_0 = torch.randn((B, *self.backbone.data_shape), device=device)

        # Define the integration time grid
        # odeint will evaluate the state at exactly these points.
        s_grid = torch.linspace(0.0, 1.0, integration_steps + 1, device=device)

        self.backbone.eval()

        # ODE solvers strictly pass a scalar time `s_scalar` and the current state `z_current`
        def ode_func(s_scalar: torch.Tensor, z_current: torch.Tensor) -> torch.Tensor:
            s_tensor = s_scalar.expand(B)
            return self.backbone(s_tensor, z_current, x_cond)

        # Solve the ODE
        # The output shape will be: (len(s_grid), B, C, H, W)
        trajectory = odeint(
            func=ode_func,
            y0=z_0,
            t=s_grid,
            method=method,  # Swaps the backend mathematically: 'euler', 'rk4', 'dopri5', etc.
            options={"step_size": 1.0 / integration_steps}
            if method in ["euler", "rk4"]
            else {},
        )

        self.backbone.train()

        if return_trajectory:
            return trajectory  # Returns all steps
        else:
            return trajectory[-1]  # Returns only the final image at s=1
