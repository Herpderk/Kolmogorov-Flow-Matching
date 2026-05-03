import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kolmogorov_flow_matching.dataset import TimeseriesDataset
from kolmogorov_flow_matching.models.base import ConditionalGenerativeFramework


def visualize_frames(x: torch.Tensor | np.ndarray):
    """
    Visualizes an arbitrary sequence of Kolmogorov flow frames.

    Args:
        x (torch.Tensor or np.ndarray): The fluid data. Expected shape is
                                        [C, H, W] or [H, W], where C is the
                                        number of frames.
    """
    # 1. Extract and format the data
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
    else:
        x = np.squeeze(x)

    # 2. Standardize to 3D [frames, H, W] for consistent looping
    if x.ndim == 2:
        x = x[np.newaxis, ...]  # Add a frame dimension if it's just one image
    elif x.ndim > 3:
        raise ValueError(
            f"Expected 2D or 3D array after squeezing, got {x.ndim}D: {x.shape}"
        )

    num_frames = x.shape[0]

    # 3. Setup the plot dynamically based on the number of frames
    fig, axes = plt.subplots(1, num_frames, figsize=(3 * num_frames, 3))

    # Matplotlib returns a single Axes object if num_frames == 1, so we wrap it in a list
    if num_frames == 1:
        axes = [axes]

    # Determine global min and max for a consistent color scale across all frames
    vmin = x.min()
    vmax = x.max()

    # 4. Plot each frame
    for i in range(num_frames):
        ax = axes[i]
        im = ax.imshow(x[i], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"Frame {i + 1}")
        ax.axis("off")

    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    # cbar.set_label('Vorticity / Velocity')

    plt.suptitle(f"Kolmogorov Flow ({num_frames} frames)", y=1.05, fontsize=14)
    plt.show()


def autoregressive_generation(
    model: ConditionalGenerativeFramework,
    initial_conditions: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """
    initial_conditions: shape [1, k_frames, H, W]
    """
    model.eval()
    current_history = initial_conditions.clone()
    predictions = []

    with torch.no_grad():
        for step in range(num_steps):
            # 1. Generate the next frame based on the current history window
            next_frame = model.sample(current_history)  # Shape: [1, 1, H, W]
            predictions.append(next_frame)

            # 2. Slide the window: Drop the oldest frame, append the new prediction
            # current_history[:, 1:] takes frames 1, 2, 3 (dropping 0)
            current_history = torch.cat([current_history[:, 1:], next_frame], dim=1)

    return torch.cat(predictions, dim=1)  # Shape: [1, num_steps, H, W]


def benchmark_integration_methods(
    model, x_cond, x_target, num_steps: int, solver_method: str
):
    """
    Runs a multi-step autoregressive rollout and records the time and MSE.
    """
    target_steps = x_target.size(1)
    current_cond = x_cond.clone()
    gen_frames = []

    # Synchronize CUDA to get accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for t in range(target_steps):
        # Generate the next frame
        next_frame = model.sample(
            current_cond, num_inference_steps=num_steps, solver=solver_method
        )
        gen_frames.append(next_frame)

        # Slide the conditioning window forward
        current_cond = torch.cat([current_cond[:, 1:], next_frame.unsqueeze(1)], dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    gen_sequence = torch.stack(gen_frames, dim=1)
    mse = F.mse_loss(gen_sequence, x_target).item()
    duration = end_time - start_time

    return mse, duration


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on {device}...")

    # 1. Load the Validation Dataset
    valid_set = TimeseriesDataset(
        h5_file_path="path/to/valid.h5",  # UPDATE THIS
        k_frames=4,
        target_steps=10,  # 10-step AR rollout
        dataset_key="train/u",  # Match your previous key
    )
    valid_loader = DataLoader(valid_set, batch_size=8, shuffle=False)

    batch = next(iter(valid_loader))
    x_cond = batch["condition"].to(device)
    x_target = batch["target"].to(device)

    # 2. Load the Models
    print("Loading models...")
    # diffusion_model = DiffusionModel(...).to(device)
    # diffusion_model.load_state_dict(torch.load("diffusion_checkpoint.pt"))
    # diffusion_model.eval()

    # fm_model = FlowMatchingModel(...).to(device)
    # fm_model.load_state_dict(torch.load("fm_checkpoint.pt"))
    # fm_model.eval()

    # 3. Define the NFE targets (Multiples of 4 work best to keep RK2 and RK4 exact)
    target_nfes = [12, 24, 36, 48, 60, 100]

    results = {
        "DDIM": {"nfe": [], "mse": [], "time": []},
        "Euler": {"nfe": [], "mse": [], "time": []},
        "RK2": {"nfe": [], "mse": [], "time": []},
        "RK4": {"nfe": [], "mse": [], "time": []},
    }

    print("Starting evaluations...")
    with torch.no_grad():
        for nfe in tqdm(target_nfes, desc="Evaluating NFEs"):
            # --- Diffusion (DDIM) ---
            # DDIM steps = NFE
            # mse_ddim, time_ddim = evaluate_ar_rollout(
            #     diffusion_model, x_cond, x_target, num_steps=nfe, solver_method="ddim"
            # )
            # results["DDIM"]["nfe"].append(nfe)
            # results["DDIM"]["mse"].append(mse_ddim)
            # results["DDIM"]["time"].append(time_ddim)

            # --- Flow Matching (Euler) ---
            # Euler steps = NFE
            # mse_euler, time_euler = evaluate_ar_rollout(
            #     fm_model, x_cond, x_target, num_steps=nfe, solver_method="euler"
            # )
            # results["Euler"]["nfe"].append(nfe)
            # results["Euler"]["mse"].append(mse_euler)
            # results["Euler"]["time"].append(time_euler)

            # --- Flow Matching (RK2) ---
            # RK2 steps = NFE / 2
            rk2_steps = max(1, nfe // 2)
            actual_rk2_nfe = rk2_steps * 2

            # mse_rk2, time_rk2 = evaluate_ar_rollout(
            #     fm_model, x_cond, x_target, num_steps=rk2_steps, solver_method="rk2"
            # )
            # results["RK2"]["nfe"].append(actual_rk2_nfe)
            # results["RK2"]["mse"].append(mse_rk2)
            # results["RK2"]["time"].append(time_rk2)

            # --- Flow Matching (RK4) ---
            # RK4 steps = NFE / 4
            rk4_steps = max(1, nfe // 4)
            actual_rk4_nfe = rk4_steps * 4

            # mse_rk4, time_rk4 = evaluate_ar_rollout(
            #     fm_model, x_cond, x_target, num_steps=rk4_steps, solver_method="rk4"
            # )
            # results["RK4"]["nfe"].append(actual_rk4_nfe)
            # results["RK4"]["mse"].append(mse_rk4)
            # results["RK4"]["time"].append(time_rk4)

    # 4. Plotting the Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colors and markers for consistency
    styles = {
        "DDIM": {"color": "blue", "marker": "o"},
        "Euler": {"color": "red", "marker": "s"},
        "RK2": {"color": "orange", "marker": "v"},
        "RK4": {"color": "green", "marker": "^"},
    }

    # Plot 1: NFE vs MSE
    for method, data in results.items():
        if len(data["nfe"]) > 0:
            ax1.plot(
                data["nfe"],
                data["mse"],
                label=method,
                **styles[method],
                linestyle="-",
                linewidth=2,
            )

    ax1.set_title("10-Step AR Prediction Error vs Compute")
    ax1.set_xlabel("Number of Function Evaluations (NFE)")
    ax1.set_ylabel("Autoregressive MSE (Lower is Better)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Plot 2: NFE vs Inference Time
    for method, data in results.items():
        if len(data["nfe"]) > 0:
            ax2.plot(
                data["nfe"],
                data["time"],
                label=method,
                **styles[method],
                linestyle="-",
                linewidth=2,
            )

    ax2.set_title("Inference Speed vs Compute")
    ax2.set_xlabel("Number of Function Evaluations (NFE)")
    ax2.set_ylabel("Time for 10-Step Batch (Seconds) (Lower is Better)")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Uncomment to run:
    # run_benchmark()
    pass
