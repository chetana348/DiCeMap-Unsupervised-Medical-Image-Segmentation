import numpy as np
import sewar
import torch
import torch.nn.functional as F
from skimage.metrics import hausdorff_distance, structural_similarity
from sklearn.metrics import accuracy_score

def ssim_wrap(img1: np.ndarray, img2: np.ndarray, **kwargs) -> float:
    """
    Computes SSIM between two images.
    Assumes input shape [H, W, C] for color or [H, W] for grayscale.

    Args:
        img1: First input image.
        img2: Second input image.
        kwargs: Additional arguments for skimage SSIM.

    Returns:
        SSIM score as float.
    """
    assert img1.shape == img2.shape, "Both inputs must have the same shape."
    height, width = img1.shape[:2]

    # Determine appropriate window size
    if min(height, width) < 7:
        window = min(height, width)
        window = window - 1 if window % 2 == 0 else window
    else:
        window = None

    # Determine channel axis for multichannel input
    use_channel_axis = -1 if img1.ndim == 3 else None

    return structural_similarity(img1,
                                 img2,
                                 channel_axis=use_channel_axis,
                                 win_size=window,
                                 **kwargs)


def ssim(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Computes SSIM between two images, ensuring consistent dynamic range handling.

    Args:
        img_ref: Ground truth or reference image [H, W, C] or [H, W]
        img_test: Test image of same shape

    Returns:
        SSIM score as float
    """
    # Ensure float type if data is boolean
    if isinstance(img_ref.max(), bool):
        img_ref = img_ref.astype(np.float32)
        img_test = img_test.astype(np.float32)

    dynamic_range = float(img_ref.max() - img_ref.min())
    dynamic_range = dynamic_range if dynamic_range != 0 else 1.0

    return ssim_wrap(img_ref, img_test, data_range=dynamic_range)