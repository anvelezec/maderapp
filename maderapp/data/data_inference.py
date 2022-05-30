import os
from typing import Tuple

import numpy as np
import torch
from albumentations.core.composition import Compose as Acompose
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MaderappDatasetInference(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None) -> None:
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:

        image = Image.open(self.annotations_file[index]).convert("RGB")

        if self.transform:
            image = (
                self.transform(image=np.array(image))["image"]
                if isinstance(self.transform, Acompose)
                else self.transform(image)
            )
        else:
            image = ToTensor()(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, self.annotations_file[index]
