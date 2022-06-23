from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.core.composition import Compose
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from maderapp.data import datasets


def trainer(
    metadata: pd.DataFrame,
    img_dir: str,
    img_dir_val: str,
    model_checkpoint_dir: str,
    logs_folder_dir: str,
    model: pl.LightningModule,
    kfold: int,
    train_trans: Compose,
    val_trans: Compose,
    model_name: str,
    dataset_type: str,
    batch_size: int,
    max_epochs: int,
    validation: bool,
    device: str,
    checkpoint_callback_monitor: str,
    patches_params: dict = None
):

    class_names = sorted(metadata.iloc[:, 1].value_counts().index)
    class_names2ids = {j: i for i, j in enumerate(class_names)}
    class_ids2names = {j: i for i, j in class_names2ids.items()}

    with open(f"{model_checkpoint_dir}labels.csv", "w") as file:
        for specie, index in class_names2ids.items():
            file.write(f"{specie};{index} \n")
            
    if kfold is not None:
        print(f"training fold={kfold}")
        train_metadata = metadata[metadata.iloc[:, 2] != kfold]
        val_metadata = metadata[metadata.iloc[:, 2] == kfold]
    else:
        print(f"train:0-val:1 schema")
        train_metadata = metadata[metadata.iloc[:, 2] == 0]
        val_metadata = metadata[metadata.iloc[:, 2] == 1]

    # Creates dataset and dataloaders
    ds_train_params = {
        "img_dir":img_dir,
        "annotations_file":train_metadata,
        "class_names2ids":class_names2ids,
        "transform":train_trans,
    }

    ds_val_params = {
        "img_dir": img_dir,
        "annotations_file": val_metadata,
        "class_names2ids": class_names2ids,
        "transform": val_trans,
    }

    if dataset_type == "patches":
        for ds_params in [ds_train_params, ds_val_params]:
            if patches_params is not None:
                ds_params.update(**patches_params)
    
    train_ds = datasets[dataset_type](**ds_train_params)
    val_ds = datasets[dataset_type](**ds_val_params)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Sets logs configuration
    logger = TensorBoardLogger(f"{logs_folder_dir}/{kfold}", name=model_name)

    # Checkpointing
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor=checkpoint_callback_monitor,
        mode="min",
        dirpath=f"{model_checkpoint_dir}/{kfold}/{model_name}",
        filename="maderapp-{epoch:02d}-{val_loss:.2f}",
    )

    # Train model
    trainer = pl.Trainer(
        accelerator=device,
        gpus=torch.cuda.device_count(),
        max_epochs=max_epochs,
        check_val_every_n_epoch=2,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if validation:
        metadata = [str(path) for path in list(Path(img_dir_val).glob("*.jpg"))]
        ds = datasets["inference"](annotations_file=metadata)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        test_preds = trainer.predict(model=model, dataloaders=dl)

        with open(f"prediction_{kfold}_{model_name}.txt", "a") as file:
            for test_pred in test_preds:
                for path, pred in zip(test_pred[0], test_pred[1]):
                    file.write(f"{path}, {class_ids2names[pred]} \n")
