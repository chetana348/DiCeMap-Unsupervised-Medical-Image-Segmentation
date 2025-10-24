import os
import warnings
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

warnings.filterwarnings("ignore")


class OutputSaver(object):
    def __init__(self, save_path: str = None, random_seed: int = None) -> None:
        self.random_seed = random_seed

        # Create subdirectories for each output
        self.dir_image = os.path.join(save_path, 'image')
        self.dir_recon = os.path.join(save_path, 'recon')
        self.dir_label = os.path.join(save_path, 'label')
        self.dir_latent = os.path.join(save_path, 'latent')

        os.makedirs(self.dir_image, exist_ok=True)
        os.makedirs(self.dir_recon, exist_ok=True)
        os.makedirs(self.dir_label, exist_ok=True)
        os.makedirs(self.dir_latent, exist_ok=True)

        self.image_idx = 0

    def save(
        self,
        image_batch: torch.Tensor,
        recon_batch: torch.Tensor,
        label_true_batch: torch.Tensor,
        latent_batch: torch.Tensor,
    ) -> None:

        image_batch = image_batch.cpu().detach().numpy()
        recon_batch = recon_batch.cpu().detach().numpy()
        if label_true_batch is not None:
            label_true_batch = label_true_batch.cpu().detach().numpy()
        latent_batch = latent_batch.cpu().detach().numpy()

        # Channel-first to channel-last
        image_batch = np.moveaxis(image_batch, 1, -1)
        recon_batch = np.moveaxis(recon_batch, 1, -1)
        if label_true_batch is not None:
            label_true_batch = np.moveaxis(label_true_batch, 1, -1)
        latent_batch = np.moveaxis(latent_batch, 1, -1)

        # Remove single channels
        image_batch = squeeze_excessive_dimension(image_batch)
        recon_batch = squeeze_excessive_dimension(recon_batch)
        if label_true_batch is not None:
            label_true_batch = squeeze_excessive_dimension(label_true_batch)

        B, H, W, C = latent_batch.shape

        if label_true_batch is None:
            label_true_batch = np.empty((B, H, W))
            label_true_batch[:] = np.nan

        for i in tqdm(range(B)):
            self.save_as_tif(
                image=image_batch[i],
                recon=recon_batch[i],
                label=label_true_batch[i],
                latent=latent_batch[i]
            )
        return

    def save_as_tif(self, image: np.array, recon: np.array,
                    label: np.array, latent: np.array) -> None:
        idx_str = str(self.image_idx).zfill(5)

        # Convert to uint8 or normalize if necessary
        def to_image(arr):
            if np.isnan(arr).any():
                arr = np.nan_to_num(arr)
            if arr.dtype != np.uint8:
                arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            return Image.fromarray(arr)

        to_image(image).save(os.path.join(self.dir_image, f'image_{idx_str}.tif'))
        to_image(recon).save(os.path.join(self.dir_recon, f'recon_{idx_str}.tif'))
        to_image(label).save(os.path.join(self.dir_label, f'label_{idx_str}.tif'))

        # Save latent as float32 multi-channel
        np.save(os.path.join(self.dir_latent, f'latent_{idx_str}.npy'), latent)

        self.image_idx += 1


def squeeze_excessive_dimension(batched_data: np.array) -> np.array:
    assert len(batched_data.shape) in [3, 4]
    if len(batched_data.shape) == 4 and batched_data.shape[-1] == 1:
        return batched_data[..., 0]
    return batched_data
