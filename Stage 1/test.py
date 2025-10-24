import os
import random
import warnings
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from model1 import *
#from model_try import *
from utils.scheduler import *
from datagen import *
from utils.output_saver import *
from utils.loss import *


def test():
    device = torch.device('cuda')
    test_set, num_image_channel =  DataGen(r'T:\Labs\QMI\CK Data\PDAC\ktrans\cropped\images_8bit', mode='test')  # r'D:\PhD\Prostate\Data\GSA\sliced_im'

    # Build the model
    model = Model(
        in_ch=1,
        base_filters=16,
        rng_seed=2,
        patches_per_image=8,
        patch_dim=5
    ).to(device)
    model.load(r"C:\Users\kris83\OneDrive - The Ohio State University Wexner Medical Center\OSU Files\QML\Vote2Segment\Stage 1\out\PDAC_ktrans\500iter\pdac_ktrans.pty", device=device)

    loss_fn_recon = torch.nn.MSELoss()
    loss_fn_contrastive = ContrastiveLoss()
    output_saver = OutputSaver(save_path=r"C:\Users\kris83\OneDrive - The Ohio State University Wexner Medical Center\OSU Files\QML\Vote2Segment\Stage 1\out\PDAC_ktrans\500iter\outputs",
                               random_seed=2)

    test_loss_recon, test_loss_contrastive, test_loss = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for _, (x_test, y_test) in enumerate(test_set):
            x_test = x_test.type(torch.FloatTensor).to(device)
            z, patch_real, patch_recon, z_anchors, z_positives = model(x_test)

            loss_recon = loss_fn_recon(patch_real, patch_recon)
            loss_contrastive = loss_fn_contrastive(z_anchors, z_positives)
            loss = 0.001 * \
                loss_contrastive + (1 - 0.001) * loss_recon

            B = x_test.shape[0]
            test_loss_recon += loss_recon.item() * B
            test_loss_contrastive += loss_contrastive.item() * B
            test_loss += loss.item() * B

            # Each pixel embedding recons to a patch.
            # Here we only take the center pixel of the reconed patch and collect into a reconed image.
            B, L, H, W = z.shape
            z_for_recon = z.permute((0, 2, 3, 1)).reshape(B, H * W, L)
            patch_recon = model.decoder(z_for_recon)
            C = patch_recon.shape[2]
            P = patch_recon.shape[-1]
            patch_recon = patch_recon[:, :, :, P // 2, P // 2]
            patch_recon = patch_recon.permute((0, 2, 1)).reshape(B, C, H, W)

            output_saver.save(image_batch=x_test,
                              recon_batch=patch_recon,
                              label_true_batch=None,
                              latent_batch=z)

    test_loss_recon = test_loss_recon / len(test_set.dataset)
    test_loss_contrastive = test_loss_contrastive / len(test_set.dataset)
    test_loss = test_loss / len(test_set.dataset)


    return


if __name__ == '__main__':
    
        test()