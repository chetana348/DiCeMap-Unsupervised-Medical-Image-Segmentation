from glob import glob
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import cv2
import numpy as np
from PIL import Image

class WrappedDataset(Dataset):
    def __init__(self, base, desired_len):
        self.base = base
        self.desired_len = desired_len

    def __len__(self):
        return self.desired_len

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]

        
class Dataset(Dataset):

    def __init__(self,
                 base_path: str = r'T:\Labs\QMI\CK Data\PDAC\ktrans\cropped\images_8bit',
                 out_shape: Tuple[int] = (128, 128)):

        self.base_path = base_path
        self.npy_paths = sorted(glob('%s/%s' % (base_path, '*.tif')))
        print(len(self.npy_paths))
        #print(len(self.npy_paths))
        self.out_shape = out_shape

    def __len__(self) -> int:
        return len(self.npy_paths)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = np.array(Image.open(self.npy_paths[idx]))
        #image = np_arr['image']

        if len(image.shape) == 3:
            assert image.shape[-1] == 1
            image = image.squeeze(-1)

        # Resize to `out_shape`. Be careful with labels.
        assert len(image.shape) in [2, 3]
        if len(image.shape) == 3:
            assert image.shape[2] == 1

        resize_factor = np.array(self.out_shape) / image.shape[:2]
        dsize = np.int16(resize_factor.min() * np.float16(image.shape[:2]))
        image = cv2.resize(src=image,
                           dsize=dsize,
                           interpolation=cv2.INTER_CUBIC)
        #print(image.shape)
        #assert image.shape == self.out_shape
        image = image[None, :, :]
        #image = image.astype(np.int8)  # Normalize to [0, 1] if it's 8-bit image
        #image = torch.from_numpy(image)

        return image, np.nan



def DataGen(config, mode: str = 'train'):
   
    dataset = Dataset()

    num_image_channel = 1

    if mode == 'train':
        # Parse and normalize train:val ratio
        ratios = [7,3]
        total = sum(ratios)
        ratios = [r / total for r in ratios]

        # Compute split sizes
        dataset_len = len(dataset)
        val_size = int(ratios[1] * dataset_len)
        train_size = dataset_len - val_size

        # Random split
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(2)
        )

        # Extend train set length to ensure enough batches
        min_batch_per_epoch = 5
        desired_len = max(len(train_set), 1 * min_batch_per_epoch)

        train_set_wrapped = WrappedDataset(train_set, desired_len)

        # DataLoaders
        train_loader = DataLoader(train_set_wrapped,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=1)
        val_loader = DataLoader(val_set,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)
        return train_loader, val_loader, num_image_channel

    else:
        test_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)
        return test_loader, num_image_channel
