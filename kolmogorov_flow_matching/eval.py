import matplotlib.pyplot as plt
import numpy as np
import torch


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
