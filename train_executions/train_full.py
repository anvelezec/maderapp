import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from maderapp.model import (
    TimberEfficientNet,
    TimberEfficientNetNS,
    TimberMobileNet,
    TimberResNet,
)
from maderapp.trainer import trainer

metadata = pd.read_csv("metadata-v3.csv", header=None)

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
OUT_FEATURES = 27
NUM_EPOCHS = 250
REQUIRE_GRAD = True
VALIDATION = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = TimberMobileNet(num_classes=OUT_FEATURES)
trainer(
    metadata=metadata,
    img_dir="../data_maderapp/training-img-v3",
    img_dir_val="../data_maderapp/validation",
    model_checkpoint_dir="../data_maderapp/checkpoints-v3",
    logs_folder_dir="../data_maderapp/model_logs-v3",
    model=model,
    kfold=None,
    train_trans=train_trans,
    val_trans=val_trans,
    model_name="MobileNet",
    dataset_type="basic",
    batch_size=BATCH_SIZE,
    max_epochs=NUM_EPOCHS,
    validation=VALIDATION,
    device=DEVICE,
    checkpoint_callback_monitor="val_loss",
)
