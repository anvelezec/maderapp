import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path

from maderapp.data import MaderappDataset
from maderapp.data_inference import MaderappDatasetInference
from maderapp.timber_clasification_efficientNet import TimberEfficientNet

batch_size = 128
out_features = 25
num_epochs = 2
require_grad = True
validation = True
kfold = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

metadata = pd.read_csv("metadata.csv")
class_names = sorted(metadata.iloc[:, 1].value_counts().index)
class_names2ids = {j: i for i, j in enumerate(class_names)}
class_ids2names = {j: i for i, j in class_names2ids.items()}

train_trans = A.Compose(
    [
        A.Resize(224, 224),
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
# Creates dataset and dataloaders
train_ds = dataset = MaderappDataset(
    img_dir="training-img",
    annotations_file=metadata[metadata.iloc[:, 2] != kfold],
    transform=train_trans,
)

val_ds = dataset = MaderappDataset(
    img_dir="training-img",
    annotations_file=metadata[metadata.iloc[:, 2] == kfold],
    transform=val_trans,
)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
model = TimberEfficientNet(num_classes=out_features)

# Sets logs configuration
logger = TensorBoardLogger("model_logs", name="experiments1")

# Checkpointing
# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="val_loss",
    mode="min",
    dirpath="model_checkpoint/efficienNet",
    filename="maderapp-{epoch:02d}-{val_loss:.2f}",
)

# Train model
trainer = pl.Trainer(
    gpus=1,
    max_epochs=40,
    check_val_every_n_epoch=2,
    logger=logger,
    callbacks=[checkpoint_callback],
)

trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

if validation:
    metadata = [str(path) for path in list(Path("validation").glob("*.jpg"))]
    ds = MaderappDatasetInference(annotations_file=metadata)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    test_preds = trainer.predict(model=model, dataloaders=dl)

    with open("prediction.txt", "a") as file:
        tests_pred_hard = []
        for test_pred in test_preds:
            for path, pred in zip(test_pred[0], test_pred[1]):
                file.write(f"{path}, {class_ids2names[pred]} \n")
