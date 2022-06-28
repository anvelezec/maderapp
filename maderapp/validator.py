from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from maderapp.data.data_inference import MaderappDatasetInference



def validator(trainer, model, model_name, class_ids2names, data_path):
    metadata = [str(path) for path in list(Path(data_path).glob("*.jpg"))]

    transformation = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
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
