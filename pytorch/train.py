import timm
import torch
import albumentations as A

from torch.nn import Linear
from torch.optim import SGD
from torchvision import models
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from data import MaderappDataset
from train_loop import TrainLoop


batch_size = 2
out_features = 25
num_epochs = 2
require_grad = True
device = "cuda" if torch.cuda.is_available() else "cpu"


train_trans = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

val_trans = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
# Creates dataset and dataloaders
train_ds = dataset = MaderappDataset(
    img_dir="train-dev",
    annotations_file= ["Almendro/Almendro (1).jpg", "Almendro/Almendro (2).jpg", "Almendro/Almendro (3).jpg"],
    transform=train_trans
)
test_ds = dataset = MaderappDataset(
    img_dir="train-dev",
    annotations_file= ["Almendro/Almendro (1).jpg", "Almendro/Almendro (2).jpg", "Almendro/Almendro (3).jpg"],
    transform = val_trans
)

train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

dataloaders = {"train": train_dl, "val": test_dl}

# https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
#model = models.densenet121(pretrained=True)
model = timm.create_model('efficientnet_b0', pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = Linear(in_features=1280, out_features=25)

# Define optmizer    
optimizer = SGD(model.parameters(), lr=1e-3)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_params)


# Train-val step
train_loop = TrainLoop(
    num_clases=out_features,
    device=device
)

train_loop.train_val(
    num_epochs=num_epochs, 
    dataloaders=dataloaders, 
    model=model, 
    criterion=CrossEntropyLoss(), 
    optimizer=optimizer, 
    scheduler=None,
    save_models="saved_models" 
)


y_trues, preds = [], []
for batch, (X, y) in enumerate(test_dl):
    X = X.float().to(device)
    y_true = y

    # Sets gradients to zero
    optimizer.zero_grad()

    # Forward and backward
    with torch.set_grad_enabled(True):
        pred = model(X)
        pred = torch.argmax(pred, axis=1)
    preds += pred.tolist()
    y_trues += y_true.tolist()

preds
y_trues
