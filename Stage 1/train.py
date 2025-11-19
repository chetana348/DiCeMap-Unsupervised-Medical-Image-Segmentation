import torch
import numpy as np
import sys
from tqdm import tqdm
#from model import *
from utils.scheduler import *
from datagen import *
from utils.loss import *
import os
from model1 import *

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_data, validation_data, input_channels = DataGen('images_8bit', mode='train')
    max_epochs=50

    # Initialize model and optimizer
    model = Model(
        in_ch=input_channels,
        base_filters=16,
        rng_seed=2,
        patches_per_image=8,
        patch_dim=5
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=10,
        warmup_start_lr=1e-3 * 1e-3,
        max_epochs=max_epochs
    )

    # Loss functions
    l2_loss = torch.nn.MSELoss()
    contrastive_loss = ContrastiveLoss()

    best_val_total_loss = np.inf

    for epoch in tqdm(range(max_epochs), desc="Epoch", dynamic_ncols=True, file=sys.stdout, position=0):
        model.train()
        train_metrics = {"l2": 0.0, "contrast": 0.0, "total": 0.0}

        for _, (batch_imgs, _) in enumerate(training_data):
            batch_imgs = batch_imgs.float().to(device)
            batch_size = batch_imgs.size(0)

            _, patch_gt, patch_pred, feature_q, feature_k = model(batch_imgs)

            loss_l2 = l2_loss(patch_pred, patch_gt)
            loss_contrast = contrastive_loss(feature_q, feature_k)
            total_loss = (
                0.001 * loss_contrast +
                (1 - 0.001) * loss_l2
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_metrics["l2"] += loss_l2.item() * batch_size
            train_metrics["contrast"] += loss_contrast.item() * batch_size
            train_metrics["total"] += total_loss.item() * batch_size

        scheduler.step()

        # Normalize train losses
        num_train = len(training_data.dataset)
        for key in train_metrics:
            train_metrics[key] /= num_train

        write_log_entry(
            f"Train [{epoch+1}/{max_epochs}] | l2 Loss: {train_metrics['l2']:.3f}, "
            f"Contrast: {train_metrics['contrast']:.3f}, Total: {train_metrics['total']:.3f}",
            output_path='logs.txt',
            display=False
        )

        # Validation loop
        model.eval()
        val_metrics = {"l2": 0.0, "contrast": 0.0, "total": 0.0}

        with torch.no_grad():
            for _, (val_imgs, _) in enumerate(validation_data):
                val_imgs = val_imgs.float().to(device)
                batch_size = val_imgs.size(0)

                _, patch_gt, patch_pred, feature_q, feature_k = model(val_imgs)

                val_l2 = l2_loss(patch_pred, patch_gt)
                val_contrast = contrastive_loss(feature_q, feature_k)
                val_total = (
                    0.001 * val_contrast +
                    (1 - 0.001) * val_l2
                )

                val_metrics["l2"] += val_l2.item() * batch_size
                val_metrics["contrast"] += val_contrast.item() * batch_size
                val_metrics["total"] += val_total.item() * batch_size

        # Normalize val losses
        num_val = len(validation_data.dataset)
        for key in val_metrics:
            val_metrics[key] /= num_val

        write_log_entry(
            f"Validation [{epoch+1}/{max_epochs}] | l2 Loss: {val_metrics['l2']:.3f}, "
            f"Contrast: {val_metrics['contrast']:.3f}, Total: {val_metrics['total']:.3f}",
            output_path='logs.txt',
            display=False
        )

        # Save best model
        if val_metrics["total"] < best_val_total_loss:
            best_val_total_loss = val_metrics["total"]
            model.save('pdac_ktrans.pty')
            write_log_entry("Model weights successfully saved.", output_path=r'logs.txt', display=False)

        tqdm.write(
            f"Epoch [{epoch+1}/{max_epochs}] | "
            f"Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}"
        )

    return




def write_log_entry(message: str, output_path: str = None, display: bool = True) -> None:
    """
    Logs a message to console and/or a specified file path.

    Args:
        message (str): The content to be logged.
        output_path (str, optional): Path to the log file.
        display (bool): Whether to print the message to stdout.
    """
    if display:
        print(message)

    if output_path is not None:
        dir_path = os.path.dirname(output_path)
        os.makedirs(dir_path, exist_ok=True)

        mode = 'a+' if os.path.exists(output_path) else 'w+'
        with open(output_path, mode) as file_obj:
            file_obj.write(message + '\n')

    
if __name__ == '__main__':

    seed = 2
        #test(config=config)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    train()
