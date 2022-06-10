import albumentations as A
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from maderapp.data_inference import MaderappDatasetInference
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2


def validator(trainer, model, model_name, class_ids2names):
    metadata = [str(path) for path in list(Path("validation").glob("*.jpg"))]

    transformation = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
            ToTensorV2(),
        ]
    )

    ds = MaderappDatasetInference(annotations_file=metadata, transform=transformation)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    test_preds = trainer.predict(model=model, dataloaders=dl)

    with open(f"results/prediction_{model_name}.txt", "a") as file:
        for test_pred in test_preds:
            for path, pred in zip(test_pred[0], test_pred[1]):
                file.write(f"{path}, {class_ids2names[pred]} \n")


if __name__ == "__main__":
    from maderapp.timber_clasification_efficientNetNS import TimberEfficientNetNS
    from maderapp.timber_clasification_mobileNet import TimberMobileNet
    from maderapp.timber_clasification_resNet import TimberResNet

    model = TimberResNet(25)
    model = model.load_from_checkpoint(
        "model_checkpoint/3/RestNet/maderapp-epoch=249-val_loss=0.05.ckpt",
        num_classes=25,
    )
    model.eval()

    trainer = pl.Trainer(gpus=1,)

    metadata = pd.read_csv("metadata.csv", header=None)
    class_names = sorted(metadata.iloc[:, 1].value_counts().index)
    class_names2ids = {j: i for i, j in enumerate(class_names)}
    class_ids2names = {j: i for i, j in class_names2ids.items()}

    validator(trainer, model, "RestNet", class_ids2names)
