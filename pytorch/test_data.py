from torch.utils.data import DataLoader

from data import MaderappDataset


dataset = MaderappDataset(
    img_dir="train-dev",
    annotations_file= ["Almendro/Almendro (1).jpg", "Almendro/Almendro (2).jpg", "Almendro/Almendro (3).jpg"],
)

dataloader = DataLoader(dataset, 2)

x = next(iter(dataloader))
x