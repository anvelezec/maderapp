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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    training_data = MaderAppDataset(img_dir="training/src/data", img_size=500)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
    train_dataloader

    images = next(iter(train_dataloader))
    train_features, train_labels = images["image"], images["label"]
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze().permute(1,2,0)
    label = train_labels[0]
    plt.imshow(img)
    plt.show()
    print(f"Label: {label}")