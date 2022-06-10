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
from maderapp.timber_clasification_efficientNetNS import TimberEfficientNetNS
from maderapp.timber_clasification_resNet import TimberResNet
from maderapp.timber_clasification_mobileNet import TimberMobileNet

batch_size = 128
out_features = 25
num_epochs = 2
require_grad = True
validation = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def main_trainer(metadata, model, kfold, train_trans, val_trans, model_name):

    print(f"training fold={kfold}")

    # Creates dataset and dataloaders
    train_metadata = metadata[metadata.iloc[:, 2] != kfold] if kfold else metadata
    train_ds = dataset = MaderappDataset(
        img_dir="training-img", annotations_file=train_metadata, transform=train_trans,
    )

    val_metadata = metadata[metadata.iloc[:, 2] == kfold] if kfold else metadata
    val_ds = dataset = MaderappDataset(
        img_dir="training-img", annotations_file=val_metadata, transform=val_trans,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Sets logs configuration
    logger = TensorBoardLogger(f"model_logs/{kfold}", name=model_name)

    # Checkpointing
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=f"model_checkpoint/{kfold}/{model_name}",
        filename="maderapp-{epoch:02d}-{val_loss:.2f}",
    )

    # Train model
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=15,
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

        with open(f"prediction_{kfold}_{model_name}.txt", "a") as file:
            for test_pred in test_preds:
                for path, pred in zip(test_pred[0], test_pred[1]):
                    file.write(f"{path}, {class_ids2names[pred]} \n")


if __name__ == "__main__":
    metadata = pd.read_csv("metadata.csv", header=None)
    class_names = sorted(metadata.iloc[:, 1].value_counts().index)
    class_names2ids = {j: i for i, j in enumerate(class_names)}
    class_ids2names = {j: i for i, j in class_names2ids.items()}

    train_trans = A.Compose(
        [
            A.Resize(224, 224),
            A.RandomCrop(width=224, height=224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )

    val_trans = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )

    """    
    for kfold in range(4):
        # Load model
        model = TimberMobileNet(num_classes=out_features)
        main_trainer(metadata, model, kfold, train_trans, val_trans, model_name="MobileNet")
    """

    from maderapp.timber_clasification_efficientNet import TimberEfficientNet
    from maderapp.timber_clasification_efficientNetNS import TimberEfficientNetNS
    from maderapp.timber_clasification_mobileNet import TimberMobileNet

    for model_class, model_name in zip(
        [TimberEfficientNet, TimberEfficientNetNS, TimberMobileNet],
        ["efficientNet_15", "efficientNet-NS_15", "MobileNet_15"],
    ):
        model = model_class(num_classes=out_features)
        main_trainer(metadata, model, None, train_trans, val_trans, model_name)
