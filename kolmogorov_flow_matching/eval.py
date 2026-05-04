import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import clear_output, display
from torch.utils.data import DataLoader
from tqdm import tqdm

from kolmogorov_flow_matching.dataset import TimeseriesDataset
from kolmogorov_flow_matching.models.diffusion import ConditionalDiffusion
from kolmogorov_flow_matching.models.flow_matching import ConditionalFlowMatching


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

    # Standardize to 3D [frames, H, W] for consistent looping
    if x.ndim == 2:
        x = x[np.newaxis, ...]  # Add a frame dimension if it's just one image
    elif x.ndim > 3:
        raise ValueError(
            f"Expected 2D or 3D array after squeezing, got {x.ndim}D: {x.shape}"
        )

    num_frames = x.shape[0]

    # Setup the plot dynamically based on the number of frames
    fig, axes = plt.subplots(1, num_frames, figsize=(3 * num_frames, 3))

    if num_frames == 1:
        axes = [axes]

    # Determine global min and max for a consistent color scale across all frames
    vmin = x.min()
    vmax = x.max()

    # Plot each frame
    for i in range(num_frames):
        ax = axes[i]
        im = ax.imshow(x[i], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"Frame {i + 1}")
        ax.axis("off")

    plt.suptitle(f"Kolmogorov Flow ({num_frames} frames)", y=1.05, fontsize=14)
    plt.show()


def evaluate_ar_rollout(model, x_cond, x_target, nfe_steps, solver_method):
    """
    Helper function to run autoregressive generation using the class method.
    """
    # The number of frames we need to generate to match the target
    ar_frames_to_predict = x_target.size(1)

    # Call the inherited AR method, routing the correct kwargs for the solver
    if solver_method == "ddim":
        gen_frames = model.autoregressive_generation(
            init_conds=x_cond,
            num_steps=ar_frames_to_predict,
            num_inference_steps=nfe_steps,  # Kwarg for Diffusion
        )
    else:
        gen_frames = model.autoregressive_generation(
            init_conds=x_cond,
            num_steps=ar_frames_to_predict,
            method=solver_method,  # Kwarg for Flow Matching
            integration_steps=nfe_steps,  # Kwarg for Flow Matching
        )

    # Calculate MSE across the entire generated trajectory
    mse = F.mse_loss(gen_frames, x_target).item()
    return mse


def benchmark_integration_methods(
    valid_set,
    diff_model,
    fm_model,
    batch_size=16,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    print(f"Running benchmark on {device}...")

    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    batch = next(iter(valid_loader))
    x_cond = batch["condition"].to(device)
    x_target = batch["target"].to(device)

    # Infer the number of AR steps from the target sequence dimension
    ar_steps = x_target.size(1)

    # Define the NFE targets
    target_nfes = torch.arange(0, 40, 4).long().tolist()[1:]

    results = {
        "DDIM": {"nfe": [], "mse": []},
        "Euler": {"nfe": [], "mse": []},
        "RK4": {"nfe": [], "mse": []},
    }

    # Colors and markers for consistency
    styles = {
        "DDIM": {"color": "blue", "marker": "o"},
        "Euler": {"color": "red", "marker": "s"},
        "RK4": {"color": "green", "marker": "^"},
    }

    # Set up the figure once outside the loop (adjusted width for a single plot)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    print(f"Starting evaluations for {ar_steps}-step autoregressive rollouts...")
    with torch.no_grad():
        for nfe_raw in tqdm(target_nfes, desc="Evaluating NFEs"):
            # Cast the requested NFE to an integer so we plot exactly what the solver uses
            nfe_int = int(nfe_raw)
            if nfe_int == 0:
                continue  # Skip 0 to avoid log scale errors

            mse_ddim = evaluate_ar_rollout(
                diff_model, x_cond, x_target, nfe_steps=nfe_int, solver_method="ddim"
            )
            results["DDIM"]["nfe"].append(nfe_int)
            results["DDIM"]["mse"].append(mse_ddim)

            mse_euler = evaluate_ar_rollout(
                fm_model, x_cond, x_target, nfe_steps=nfe_int, solver_method="euler"
            )
            # Store the integer nfe_int
            results["Euler"]["nfe"].append(nfe_int)
            results["Euler"]["mse"].append(mse_euler)

            # RK4 uses 4 evaluations per step
            rk4_steps = max(1, nfe_int // 4)
            actual_rk4_nfe = rk4_steps * 4

            mse_rk4 = evaluate_ar_rollout(
                fm_model, x_cond, x_target, nfe_steps=rk4_steps, solver_method="rk4"
            )
            results["RK4"]["nfe"].append(actual_rk4_nfe)
            results["RK4"]["mse"].append(mse_rk4)

            # --- LIVE PLOTTING UPDATE ---
            ax.clear()

            # Plot: NFE vs MSE
            for method, data in results.items():
                if len(data["nfe"]) > 0:
                    ax.plot(
                        data["nfe"],
                        data["mse"],
                        label=method,
                        **styles[method],
                        linestyle="-",
                        linewidth=2,
                    )

            ax.set_title(f"{ar_steps}-Step AR MSE vs. NFE")
            ax.set_xlabel("Number of Function Evaluations (NFE)")
            ax.set_ylabel("Autoregressive MSE (Lower is Better)")
            # ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.7)

            # Position legend on the middle right (outside the plot)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

            clear_output(wait=True)
            display(fig)

    plt.savefig("benchmark_results.png", dpi=300)
    plt.close(fig)
