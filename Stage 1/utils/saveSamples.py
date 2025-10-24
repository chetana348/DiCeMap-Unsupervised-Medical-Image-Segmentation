import os
import warnings
from typing import Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


class saveSamples:
    """
    Stores batch-wise model outputs (images, reconstructions, labels, features) into .npz files.

    Args:
        output_path: Base directory to store the result files.
        random_seed: Optional seed for reproducibility (not directly used here).
    """

    def __init__(self, output_path: str, random_seed: Optional[int] = None) -> None:
        self.seed = random_seed
        self.output_dir = os.path.join(output_path, 'numpy_files')
        os.makedirs(self.output_dir, exist_ok=True)
        self.counter = 0

    def save_batch(
        self,
        inputs: torch.Tensor,
        recons: torch.Tensor,
        labels: Optional[torch.Tensor],
        features: torch.Tensor,
    ) -> None:
        """
        Saves each sample in the batch as a .npz file.

        Args:
            inputs: Original images [B, C, H, W]
            recons: Reconstructed outputs [B, C, H, W]
            labels: Ground truth labels [B, 1, H, W] or None
            features: Latent maps [B, C, H, W]
        """

        inputs_np = inputs.cpu().numpy()
        recons_np = recons.cpu().numpy()
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy() if labels is not None else None

        # Convert to [B, H, W, C]
        inputs_np = np.moveaxis(inputs_np, 1, -1)
        recons_np = np.moveaxis(recons_np, 1, -1)
        features_np = np.moveaxis(features_np, 1, -1)
        if labels_np is not None:
            labels_np = np.moveaxis(labels_np, 1, -1)

        # Remove singleton channels
        inputs_np = _reshape(inputs_np)
        recons_np = _reshape(recons_np)
        if labels_np is not None:
            labels_np = _reshape(labels_np)

        batch_size, height, width, depth = features_np.shape

        # If label is None, use nan mask
        if labels_np is None:
            labels_np = np.full((batch_size, height, width), fill_value=np.nan)

        # Save each sample individually
        for i in tqdm(range(batch_size)):
            self._save_npz_file(
                image=inputs_np[i],
                recon=recons_np[i],
                label=labels_np[i],
                feature_map=features_np[i].reshape(height * width, depth)
            )

    def _save_npz_file(self, image: np.ndarray, recon: np.ndarray,
                       label: np.ndarray, feature_map: np.ndarray) -> None:
        filename = f"sample_{str(self.counter).zfill(5)}.npz"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb+') as f:
            np.savez(f, image=image, recon=recon, label=label, latent=feature_map)
        self.counter += 1


def _reshape(batch_arr: np.ndarray) -> np.ndarray:
    """
    Removes trailing singleton channels from batched data, if present.

    Args:
        batch_arr: Array of shape [B, H, W, C] or [B, H, W]

    Returns:
        Squeezed array of shape [B, H, W] if C == 1, else unchanged
    """
    assert batch_arr.ndim in (3, 4), "Expected 3D or 4D array"
    if batch_arr.ndim == 4 and batch_arr.shape[-1] == 1:
        return batch_arr.reshape(batch_arr.shape[:3])
    return batch_arr
