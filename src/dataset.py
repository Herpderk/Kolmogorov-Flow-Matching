import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_hdf5_stats(
    h5_file_path: str, dataset_key: str = "u", chunk_size: int = 100
):
    """
    Computes global mean and std of a massive HDF5 dataset lazily.
    """
    with h5py.File(h5_file_path, "r") as f:
        data = f[dataset_key]
        num_trajectories = data.shape[0]

        sum_val = 0.0
        sq_sum_val = 0.0
        total_elements = 0

        # Iterate over trajectories in manageable chunks
        for i in range(0, num_trajectories, chunk_size):
            chunk = data[i : i + chunk_size][...]  # Read chunk into memory
            sum_val += chunk.sum()
            sq_sum_val += (chunk**2).sum()
            total_elements += chunk.size

        mean = sum_val / total_elements
        variance = (sq_sum_val / total_elements) - (mean**2)
        std = np.sqrt(variance)

    return float(mean), float(std)


class NormalizedKolmogorovDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        k_frames: int = 4,
        dataset_key: str = "u",
        mean: float = None,
        std: float = None,
    ):
        super().__init__()
        self.k_frames = k_frames
        self.h5_file_path = h5_file_path
        self.dataset_key = dataset_key

        # 1. Handle Automatic Normalization
        if mean is None or std is None:
            print(f"Computing global statistics for {h5_file_path}...")
            self.mean, self.std = compute_hdf5_stats(h5_file_path, dataset_key)
            print(f"Done! Mean: {self.mean:.5f}, Std: {self.std:.5f}")
        else:
            self.mean = mean
            self.std = std

        # 2. Get dataset shapes
        with h5py.File(self.h5_file_path, "r") as f:
            self.shape = f[self.dataset_key].shape

        self.num_trajectories = self.shape[0]
        self.time_steps = self.shape[1]

        self.samples_per_traj = self.time_steps - self.k_frames
        self.total_samples = self.num_trajectories * self.samples_per_traj

        self.data_handle = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.data_handle is None:
            self.data_handle = h5py.File(self.h5_file_path, "r")[self.dataset_key]

        traj_idx = idx // self.samples_per_traj
        t_start = idx % self.samples_per_traj
        t_target = t_start + self.k_frames

        history = self.data_handle[traj_idx, t_start:t_target]
        target = self.data_handle[traj_idx, t_target]

        # Convert to PyTorch tensors
        x_cond = torch.from_numpy(history).float()
        x_target = torch.from_numpy(target).float().unsqueeze(0)

        # 3. APPLY NORMALIZATION
        # (x - mean) / std ensures the network sees nicely scaled inputs
        x_cond = (x_cond - self.mean) / self.std
        x_target = (x_target - self.mean) / self.std

        return {"condition": x_cond, "target": x_target}

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Call this on your model's outputs during inference to convert
        them back to standard physical Kolmogorov flow values.
        """
        return (tensor * self.std) + self.mean
