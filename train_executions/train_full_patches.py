import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from maderapp.model import (
    TimberEfficientNet,
    TimberEfficientNetNS,
    TimberMobileNet,
    TimberResNet,
    TimberPatchesNet
)
from maderapp.trainer import trainer

metadata = pd.read_csv("metadata.csv", header=None)

BATCH_SIZE = 128
OUT_FEATURES = 27
NUM_EPOCHS = 250
REQUIRE_GRAD = True
VALIDATION = True
MAX_EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCHES_KERNEL = 64

train_trans = A.Compose(
    [
        A.Resize(PATCHES_KERNEL, PATCHES_KERNEL),
        A.RandomCrop(width=PATCHES_KERNEL, height=PATCHES_KERNEL),
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
        A.Resize(PATCHES_KERNEL, PATCHES_KERNEL),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
)

model = TimberPatchesNet(num_classes=OUT_FEATURES, patches_kernel=PATCHES_KERNEL)
trainer(
    metadata=metadata,
    img_dir="../data_maderapp/training-img-v3",
    img_dir_val="../data_maderapp/validation",
    model_checkpoint_dir="../data_maderapp/checkpoints-v3p",
    logs_folder_dir="../data_madetapp/model_logs-v3p",
    model=model,
    kfold=None,
    train_trans=train_trans,
    val_trans=val_trans,
    model_name="PatchesNet",
    dataset_type="patches",
    batch_size=BATCH_SIZE,
    max_epochs=NUM_EPOCHS,
    validation=VALIDATION,
    device=DEVICE,
    checkpoint_callback_monitor="val_loss",
    patches_params={"patches_kernel": PATCHES_KERNEL, "image_size": 224},
)
