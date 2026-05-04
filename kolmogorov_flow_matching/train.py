import torch
import torch.nn.functional as F
import wandb
from IPython.display import display
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from kolmogorov_flow_matching.dataset import (
    TimeseriesDataset,
)
from kolmogorov_flow_matching.models.base import ConditionalGenerativeFramework

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


def train_conditional_generative_model(
    model: ConditionalGenerativeFramework,
    train_set: TimeseriesDataset,
    valid_set: TimeseriesDataset,
    optimizer: Optimizer,
    num_epochs: int,
    batch_size: int,
    max_grad_norm: float,
    num_val_batches: int = 4,
    log_interval: int = 400,
    val_interval: int = 4000,
    num_workers: int = 4,
    persistent_workers: bool = True,
    use_wandb: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shuffle=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        shuffle=True,
        pin_memory=True,
    )

    print(f"Starting training on {device} (WandB: {use_wandb})")
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        # position=0 ensures the training bar stays at the top
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                x_cond = batch["condition"].to(device)
                x_target = batch["target"].to(device)
                if x_target.ndim == 5:
                    x_target = x_target[:, 0]

                optimizer.zero_grad()
                loss = model.get_training_loss(x_target, x_cond)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

                global_step += 1
                current_loss = loss.item()
                progress_bar.set_postfix({"Loss": f"{current_loss:.5f}"})

                if global_step % log_interval == 0:
                    samples_seen = global_step * batch_size  # Calculate samples
                    if use_wandb and wandb.run is not None:
                        wandb.log(
                            {
                                "train/loss": current_loss,
                                "train/epoch": epoch + (batch_idx / len(train_loader)),
                                "train/samples": samples_seen,  # Log as a separate metric
                                "global_step": global_step,
                            },
                            step=global_step,
                        )

                if global_step == 1 or global_step % val_interval == 0:
                    validate(
                        model,
                        valid_loader,
                        device,
                        global_step,
                        use_wandb,
                        max_batches=num_val_batches,
                    )
                    model.train()


def validate(model, valid_loader, device, global_step, use_wandb, max_batches=None):
    model.eval()
    v_loss_total = 0.0
    ar_loss_total = 0.0
    batches_processed = 0
    viz_data = None

    total_val_steps = max_batches if max_batches is not None else len(valid_loader)

    with torch.no_grad():
        val_bar = tqdm(
            enumerate(valid_loader),
            total=total_val_steps,
            desc=f"Validating [Step {global_step}]",
            leave=False,
            position=1,
        )

        for i, v_batch in val_bar:
            if max_batches is not None and i >= max_batches:
                break

            v_cond = v_batch["condition"].to(device)
            v_target = v_batch["target"].to(device)

            if i == 0:
                viz_data = {"condition": v_cond[0:1], "target": v_target[0:1]}

            # 1-step Loss
            v_loss_total += model.get_training_loss(v_target[:, 0], v_cond).item()

            # Autoregressive Rollout
            curr_cond = v_cond.clone()
            gen_frames = []
            for t in range(v_target.size(1)):
                out = model.sample(curr_cond)
                gen_frames.append(out)
                curr_cond = torch.cat([curr_cond[:, 1:], out], dim=1)

            current_ar_batch_loss = F.mse_loss(
                torch.stack(gen_frames, dim=1), v_target
            ).item()
            ar_loss_total += current_ar_batch_loss
            batches_processed += 1

            val_bar.set_postfix({"AR-MSE": f"{ar_loss_total / batches_processed:.5f}"})

        val_bar.close()  # Clean up the bar from the display

    avg_v = v_loss_total / batches_processed
    avg_ar = ar_loss_total / batches_processed

    x_gen_eval = model.sample(viz_data["condition"])
    target_f = viz_data["target"][0, 0]
    gen_f = x_gen_eval[0]
    residual = gen_f - target_f

    residual_vis = (residual + 1.0) / 2.0
    comparison_strip = torch.cat([target_f, gen_f, residual_vis], dim=2)

    if use_wandb and wandb.run is not None:
        wandb.log(
            {
                "val/MSE_one_step": avg_v,
                "val/MSE_ar_rollout": avg_ar,
                "val/comparison_strip": wandb.Image(
                    comparison_strip,
                    caption=f"Step {global_step} | Target | Gen | Residual(Biased)",
                ),
                "global_step": global_step,
            },
            step=global_step,
        )
    else:
        # Local summary output
        print(
            f"\r[Step {global_step}] Val Summary - 1-Step: {avg_v:.5f}, AR Rollout: {avg_ar:.5f}"
        )
        try:
            img = ToPILImage()(comparison_strip.cpu().clamp(0, 1))
            display(img)
        except Exception as e:
            print(f"Could not display image: {e}")

    return avg_v, avg_ar


def save_checkpoint(
    save_path: str,
    model: ConditionalGenerativeFramework,
    optimizer: Optimizer,
    num_epochs: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
        },
        save_path,
    )
    print(f"Checkpoint saved to {save_path}")
