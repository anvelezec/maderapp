import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import Conv2d, LeakyReLU, Linear, MaxPool2d, Module, ReLU, Sequential
from torch.optim import StepLR


class ResidualBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        self.act = LeakyReLU()
        self.model = Sequential(self.conv, self.act, self.conv)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.model(x)
        x = torch.add(x, shortcut)
        return x


class ActPool(Module):
    def __init__(
        self, negative_slope: float = 0.1, max_pool_kernel_size: int = 2
    ) -> None:
        super().__init__()
        self.act = LeakyReLU(negative_slope)
        self.max_pool = MaxPool2d(max_pool_kernel_size)
        self.model = Sequential(self.act, self.max_pool)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


class Flatten(Module):
    def __init__(self, start_dim: int, end_dim: int) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor):
        return torch.flatten(x, self.start_dim, self.end_dim)


class LinearBlock(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.linear1 = Linear(in_features=in_features, out_features=256)
        self.linear2 = Linear(in_features=256, out_features=out_features)
        self.relu = ReLU()
        self.model = Sequential(self.linear1, self.relu, self.linear2)

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x


class TimberPatchesNet(pl.LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.train_f1score = torchmetrics.F1Score(num_classes=num_classes)
        self.test_f1score = torchmetrics.F1Score(num_classes=num_classes)

        self.residual_block = ResidualBlock(
            in_channels=64, out_channels=64, kernel_size=3
        )
        self.act_pool = ActPool(negative_slope=0.1, max_pool_kernel_size=2)
        self.flatten = Flatten(start_dim=1, end_dim=-1)
        conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.linear = LinearBlock(in_features=1024, out_features=num_classes)

        model = [conv]
        for _ in range(4):
            model.extend([self.residual_block, self.act_pool])

        model.extend([self.flatten, self.linear])
        self.model = Sequential(*model)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        out = self.softmax(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.1)
        return optimizer, scheduler

    def step(self, batch, mode: str):
        x, y = batch[0], batch[1]
        y_hat = self.model(x)

        loss = F.cross_entropy(y_hat, y)
        acc = self.train_acc(y_hat, y)
        f1score = self.train_f1score(y_hat, y)

        self.log(
            f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        ),
        self.log(
            f"{mode}_f1score",
            f1score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        ),
        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.step(batch=batch, mode="train")
        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        loss = self.step(batch=batch, mode="test")
        return loss
