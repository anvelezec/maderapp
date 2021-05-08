import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize
from torchvision.io import read_image

from pathlib import Path

class MaderAppDataset(Dataset):
    def __init__(self, img_dir, img_size, transform=None, target_transform=None):
        imgs = Path(img_dir).glob("*/*")
        self.imgs = list(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.resize = Resize((img_size, img_size))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = str(self.imgs[idx])
        image = read_image(img_path)
        image = self.resize(image)
        label = int(self.imgs[idx].parts[-2])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
