import os
from typing import Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.core.composition import Compose as Acompose
from albumentations.pytorch import ToTensorV2
from matplotlib import transforms
from pandas import array
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from maderapp.utils import extract_patches


class MaderappPatchesDataset(Dataset):
    def __init__(
        self,
        img_dir,
        annotations_file,
        class_names2ids,
        patches_kernel,
        patches_shuffle=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform
        self.class_names2ids = class_names2ids
        self.patches_kernel = patches_kernel
        self.patches_shuffle = patches_shuffle

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index) -> None:
        img_metadata_path = os.path.join(
            self.img_dir, self.annotations_file.iloc[index, 0]
        )

        image = Image.open(img_metadata_path).convert("RGB")
        tranformation = A.Compose([A.Resize(224, 224), ToTensorV2()])
        image = tranformation(image=np.asarray(image))["image"]
        images = extract_patches(
            image=image.unsqueeze(dim=0),
            channel=3,
            kernel_height=self.patches_kernel,
            kernel_width=self.patches_kernel,
        )

        label_name = self.annotations_file.iloc[index, 1]
        label = self.class_names2ids[label_name]
        labels = [label for _ in range(images.shape[0])]

        patch_images = []
        if self.transform:
            for idx in range(images.shape[0]):
                image = np.array(images[idx]).transpose(1, 2, 0)
                image = (
                    self.transform(image=np.array(image))["image"]
                    if isinstance(self.transform, Acompose)
                    else self.transform(image)
                )
                patch_images.append(image.unsqueeze(dim=0))
            patch_images = torch.concat(patch_images, dim=0)

            if self.patches_shuffle:
                idx = torch.randperm(patch_images.shape[0])
                patch_images = patch_images[idx, :]

        if self.target_transform:
            labels = self.target_transform(labels)

        return patch_images, torch.Tensor(labels)
