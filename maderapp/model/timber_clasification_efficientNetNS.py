import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

# Use forward for inference (predicting).
# Use training_step for training.


class TimberEfficientNetNS(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.train_f1score = torchmetrics.F1Score(num_classes=num_classes)
        self.test_f1score = torchmetrics.F1Score(num_classes=num_classes)
        self.softmax = torch.nn.Softmax()

        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        trainable_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        notrainable_total_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )

        print(
            f"Number no trainable parameters: {notrainable_total_params} \nNumber trainable parameters: {trainable_total_params}"
        )

    def forward(self, x):
        out = self.model(x)
        return self.softmax(out)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        # logs metrics for each training_step - [default:True],
        # the average across the epoch, to the progress bar and logger-[default:False]
        acc = self.train_acc(y_hat, y)
        f1score = self.train_f1score(y_hat, y)

        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        ),
        self.log(
            "train_f1score",
            f1score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        ),
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.test_acc(y_hat, y)
        f1score = self.test_f1score(y_hat, y)

        # logs metrics for each validation_step - [default:False]
        # the average across the epoch - [default:True]
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_f1score", f1score, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx: int):
        x, path = batch[0], batch[1]
        pred = self.model(x)
        return path, torch.argmax(pred, axis=1).tolist()
