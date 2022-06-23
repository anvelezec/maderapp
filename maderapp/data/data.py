import os
from typing import Tuple

import numpy as np
import torch
from albumentations.core.composition import Compose as Acompose
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MaderappDataset(Dataset):
    def __init__(
        self,
        img_dir,
        annotations_file,
        class_names2ids,
        transform=None,
        target_transform=None,
    ) -> None:
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform
        self.class_names2ids = class_names2ids

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:

        img_metadata_path = os.path.join(
            self.img_dir, self.annotations_file.iloc[index, 0]
        )

        try:
            image = Image.open(img_metadata_path).convert("RGB")
            label_name = self.annotations_file.iloc[index, 1]
            label = self.class_names2ids[label_name]

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

            return image, label

        except OSError as e:
            print(f"file with error {img_metadata_path}")
            raise(Exception(print(e)))
