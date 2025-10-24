import os
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import random
from utils.metrics import ssim


########################################
# 🔹 Attention Modules
########################################

class ECABlock(nn.Module):
    """Efficient Channel Attention (lightweight)"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention via 1D conv
        y = self.avg_pool(x)                      # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))   # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


class SpatialAttentionLite(nn.Module):
    """Lightweight spatial attention via depthwise convolution"""
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(1, 1,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spatial attention on average-pooled channel map
        avg_out = torch.mean(x, dim=1, keepdim=True)   # [B, 1, H, W]
        attn = self.sigmoid(self.conv(avg_out))        # [B, 1, H, W]
        return x * attn


class LightweightAttentionBlock(nn.Module):
    """Combines ECA (channel) + lightweight spatial attention"""
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ECABlock(channels)
        self.spatial_attn = SpatialAttentionLite()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


########################################
# 🔹 Model Components
########################################

class Model(nn.Module):
    def __init__(self, in_ch: int = 1,
                 base_filters: int = 16,
                 rng_seed: int = 2,
                 patches_per_image: int = 8,
                 patch_dim: int = 5) -> None:
        super().__init__()

        self.input_channels = in_ch
        self.patch_dim = patch_dim
        self.feature_depth = base_filters * 8

        # ✅ Encoder with attention at each stage
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_filters, kernel_size=5,
                      padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(),
            LightweightAttentionBlock(base_filters),

            nn.Conv2d(base_filters, base_filters * 2, kernel_size=5,
                      padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(),
            LightweightAttentionBlock(base_filters * 2),

            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=5,
                      padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(),
            LightweightAttentionBlock(base_filters * 4),

            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=5,
                      padding='same', padding_mode='replicate'),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(),
            LightweightAttentionBlock(base_filters * 8)
        )

        # ✅ Query sampler & decoder stay same
        self.queries = QueryExtract(
            seed=rng_seed,
            patch_dim=patch_dim,
            patches_per_image=patches_per_image
        )

        self.decoder = Decoder(
            input_channels=in_ch,
            patch_dim=patch_dim,
            feature_dim=self.feature_depth
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(path, map_location=device))

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        bsz, ch, h, w = img.shape

        feat_map = self.encoder(img)  # Feature extraction w/ attention

        sem_pix, pos_pix = self.queries(img)
        num_samples = sem_pix.size(1)

        raw_patches = torch.zeros((bsz, num_samples, ch,
                                   self.patch_dim, self.patch_dim),
                                   device=img.device)
        sem_queries = torch.zeros((bsz, num_samples, self.feature_depth),
                                  device=img.device)
        target_queries = torch.zeros_like(sem_queries)

        half_patch = self.patch_dim // 2

        for i in range(bsz):
            for j in range(num_samples):
                y_sem, x_sem = sem_pix[i, j]
                y_pos, x_pos = pos_pix[i, j]

                raw_patches[i, j] = img[i, :,
                                        y_sem - half_patch:y_sem + half_patch + 1,
                                        x_sem - half_patch:x_sem + half_patch + 1]

                sem_queries[i, j] = feat_map[i, :, y_sem, x_sem]
                target_queries[i, j] = feat_map[i, :, y_pos, x_pos]

        out = self.decoder(sem_queries)

        return feat_map, raw_patches, out, sem_queries, target_queries


########################################
# 🔹 Decoder
########################################

class Decoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 patch_dim: int,
                 feature_dim: int) -> None:
        super().__init__()

        self.channels = input_channels
        self.patch_dim = patch_dim
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, input_channels * patch_dim * patch_dim),
            nn.LeakyReLU(),
            nn.Linear(input_channels * patch_dim * patch_dim,
                      input_channels * patch_dim * patch_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape [B, N, D]
        Returns:
            Tensor of shape [B, N, C, P, P]
        """
        batch_size, num_patches, _ = features.shape
        projected = self.decoder(features)  # [B, N, flat_dim]
        out = projected.view(batch_size, num_patches,
                             self.channels, self.patch_dim, self.patch_dim)
        return out


########################################
# 🔹 Query Extractor (unchanged)
########################################

class QueryExtract:
    """
    Selects query and neighbor patch coordinates for a given batch of images.
    Neighbor patches are selected based on structural similarity (SSIM).
    """

    def __init__(self,
                 seed: int = 2,
                 patch_dim: int = None,
                 patches_per_image: int = None) -> None:
        self.seed = seed
        self.patch_dim = patch_dim
        self.patches_per_image = patches_per_image

        self.max_trials = 20
        self.similarity_threshold = 0.5

    def __call__(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = images.shape

        queries = np.zeros((batch_size, self.patches_per_image, 2), dtype=int)
        neighbors = np.zeros_like(queries)

        h_margin = (self.patch_dim // 2, height - self.patch_dim // 2)
        w_margin = (self.patch_dim // 2, width - self.patch_dim // 2)

        random.seed(self.seed)

        for b in range(batch_size):
            for i in range(self.patches_per_image):
                h = random.randrange(*h_margin)
                w = random.randrange(*w_margin)
                queries[b, i] = [h, w]

                neighbor_coord = self._find_similar_patch(
                    image=images[b],
                    query_hw=(h, w),
                    full_height=height,
                    full_width=width
                )
                neighbors[b, i] = neighbor_coord

        assert queries.shape == neighbors.shape
        return torch.tensor(queries), torch.tensor(neighbors)

    def _find_similar_patch(self,
                            image: torch.Tensor,
                            query_hw: Tuple[int, int],
                            full_height: int,
                            full_width: int) -> Tuple[int, int]:
        best_match = None

        for _ in range(self.max_trials):
            candidate = self._sample_within_locality(
                base=query_hw,
                H=full_height,
                W=full_width
            )

            similarity = self._compute_patch_similarity(
                image=image.cpu().detach().numpy(),
                coord1=query_hw,
                coord2=candidate
            )

            if similarity > self.similarity_threshold:
                best_match = candidate
                break

        if best_match is None:
            best_match = self._fallback_neighbor(query_hw, full_height, full_width)

        return best_match

    def _sample_within_locality(self,
                                base: Tuple[int, int],
                                H: int,
                                W: int,
                                radius: int = 5) -> Tuple[int, int]:
        h0 = max(base[0] - radius, self.patch_dim // 2)
        h1 = min(base[0] + radius, H - self.patch_dim // 2)

        w0 = max(base[1] - radius, self.patch_dim // 2)
        w1 = min(base[1] + radius, W - self.patch_dim // 2)

        return random.randrange(h0, h1), random.randrange(w0, w1)

    def _fallback_neighbor(self, coord: Tuple[int, int], H: int, W: int) -> Tuple[int, int]:
        h, w = coord
        if h > H // 2:
            h -= 1
        else:
            h += 1

        if w > W // 2:
            w -= 1
        else:
            w += 1

        return h, w

    def _compute_patch_similarity(self,
                                  image: np.ndarray,
                                  coord1: Tuple[int, int],
                                  coord2: Tuple[int, int]) -> float:
        """
        Computes SSIM between two patches in the image.
        Args:
            image: numpy array of shape [C, H, W]
        """
        c, h, w = image.shape
        ps = self.patch_dim
        h1, w1 = coord1
        h2, w2 = coord2

        patch1 = image[:, h1 - ps // 2:h1 + ps // 2 + 1,
                          w1 - ps // 2:w1 + ps // 2 + 1]
        patch2 = image[:, h2 - ps // 2:h2 + ps // 2 + 1,
                          w2 - ps // 2:w2 + ps // 2 + 1]

        patch1 = np.moveaxis(patch1, 0, -1)
        patch2 = np.moveaxis(patch2, 0, -1)

        return ssim(patch1, patch2)
