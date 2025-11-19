import os
import cv2
import numpy as np
from glob import glob
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from tqdm import tqdm

# Paste your existing OutputSaver class here or import if available
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


# Dataset to load images with your exact resizing and shape logic
class Dataset(Dataset):
    def __init__(self,
                 base_path: str,
                 out_shape: Tuple[int] = (128, 128)):
        self.base_path = base_path
        self.npy_paths = sorted(glob(f'{base_path}/*.tif'))
        self.out_shape = out_shape

    def __len__(self) -> int:
        return len(self.npy_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        image = np.array(Image.open(self.npy_paths[idx]))

        if len(image.shape) == 3:
            assert image.shape[-1] == 1
            image = image.squeeze(-1)

        assert len(image.shape) == 2

        # Resize while preserving aspect ratio
        resize_factor = np.array(self.out_shape) / image.shape[:2]
        dsize = np.int16(resize_factor.min() * np.float16(image.shape[:2]))
        image = cv2.resize(src=image,
                           dsize=dsize,
                           interpolation=cv2.INTER_CUBIC)

        image = image[None, :, :]  # add channel dim

        return torch.from_numpy(image).float()


# Dataset to load masks (labels) in exact same way as images
class MaskDataset(Dataset):
    def __init__(self,
                 base_path: str,
                 out_shape: Tuple[int] = (128, 128)):
        self.base_path = base_path
        self.npy_paths = sorted(glob(f'{base_path}/*.tif'))
        self.out_shape = out_shape

    def __len__(self) -> int:
        return len(self.npy_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        mask = np.array(Image.open(self.npy_paths[idx]))

        if len(mask.shape) == 3:
            assert mask.shape[-1] == 1
            mask = mask.squeeze(-1)

        assert len(mask.shape) == 2

        # Resize while preserving aspect ratio
        resize_factor = np.array(self.out_shape) / mask.shape[:2]
        dsize = np.int16(resize_factor.min() * np.float16(mask.shape[:2]))
        mask = cv2.resize(src=mask,
                          dsize=dsize,
                          interpolation=cv2.INTER_NEAREST)  # Nearest for masks!

        mask = mask[None, :, :]
        mask[mask>0] = 1;

        return torch.from_numpy(mask).float()


# Combined dataset to yield (image, mask) pairs
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, out_shape=(128, 128)):
        self.image_dataset = Dataset(image_dir, out_shape)
        self.mask_dataset = MaskDataset(mask_dir, out_shape)
        assert len(self.image_dataset) == len(self.mask_dataset), "Image and mask counts differ!"

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image = self.image_dataset[idx]
        mask = self.mask_dataset[idx]
        return image, mask


def save_labels_only(image_dir, mask_dir, save_path, out_shape=(128, 128)):
    combined_dataset = ImageMaskDataset(image_dir, mask_dir, out_shape)
    test_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False, num_workers=0)

    output_saver = OutputSaver(save_path=save_path, random_seed=2)
    output_saver.image_idx = 0

    # Dummy tensors for image, recon, latent (to satisfy OutputSaver)
    dummy_image = torch.zeros(1, 1, out_shape[0], out_shape[1])
    dummy_recon = torch.zeros_like(dummy_image)
    dummy_latent = torch.zeros(1, 16, out_shape[0], out_shape[1])  # adjust channels if needed

    for image_batch, label_batch in test_loader:
        output_saver.save(
            image_batch=dummy_image.repeat(image_batch.size(0), 1, 1, 1),
            recon_batch=dummy_recon.repeat(image_batch.size(0), 1, 1, 1),
            label_true_batch=label_batch,
            latent_batch=dummy_latent.repeat(image_batch.size(0), 1, 1, 1)
        )

    print(f"Labels saved successfully at {save_path}")


if __name__ == '__main__':
    save_labels_only(
        image_dir='sliced_im',
        mask_dir='sliced_lb',
        save_path="new",
        out_shape=(128, 128)
    )
