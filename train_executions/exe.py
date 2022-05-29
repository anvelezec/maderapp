import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from maderapp.model import (
    TimberEfficientNet,
    TimberEfficientNetNS,
    TimberMobileNet,
    TimberResNet,
)
from maderapp.trainer import trainer


metadata = pd.read_csv("metadata.csv", header=None)

train_trans = A.Compose(
    [
        A.Resize(224, 224),
        A.RandomCrop(width=224, height=224),
        A.augmentations.geometric.rotate.Rotate(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
)

val_trans = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
)

BATCH_SIZE = 128
OUT_FEATURES = 25
NUM_EPOCHS = 2
REQUIRE_GRAD = True
VALIDATION = True
MAX_EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for kfold in range(4):
    # Load model
    model = TimberMobileNet(num_classes=OUT_FEATURES)
    trainer(
        metadata,
        "../data_maderapp/training-img",
        model,
        kfold,
        train_trans,
        val_trans,
        "MobileNet",
        BATCH_SIZE,
        NUM_EPOCHS,
        VALIDATION,
        DEVICE,
        "../data_madetapp/model_logs",
    )
