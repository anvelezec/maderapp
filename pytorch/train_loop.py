from typing import Callable, Tuple
from matplotlib.pyplot import axis
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path


class TrainLoop:
    def __init__(
        self,
        num_clases: int,
        device: str
    ) -> None:
        self.num_clases = num_clases
        self.device = device

    def train_val(
        self,
        num_epochs, dataloaders: dict, model: Module, criterion: Module, optimizer: Optimizer, scheduler: _LRScheduler, save_models: str
    ):
        models_folder = Path(save_models)
        if not models_folder.exists():
            models_folder.mkdir()

        self.prev_loss = torch.tensor([float("inf")]).to(self.device)

        notrainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_total_params = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )

        print(
            f"Number no trainable parameters: {notrainable_total_params} \nNumber trainable parameters: {trainable_total_params}"
        )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # train one epoch
            self._train_step(
                model,
                dataloaders["train"],
                criterion,
                self.device,
                optimizer,
                len(dataloaders["train"].dataset)
            )

            # evaluate
            self.prev_loss = self._val_step(
                model,
                dataloaders["val"],
                criterion,
                self.device,
                models_folder,
                epoch,
                self.prev_loss,
                len(dataloaders["val"].dataset)
            )

            # https://discuss.pytorch.org/t/how-to-use-torch-optim-lr-scheduler-exponentiallr/12444
            if scheduler is not None:
                scheduler.step()
        

        print("Done!")

    @staticmethod
    def _train_step(
        model: Module,
        data_loader: DataLoader,
        loss_fun: Callable,
        device: str,
        optimizer: Optimizer,
        dl_size
    ):

        running_loss, running_acc = 0, 0
        model.train()
        for batch, (X, y) in enumerate(data_loader):
            y = y.to(device)
            X = X.float().to(device)

            # Sets gradients to zero
            optimizer.zero_grad()

            # Forward and backward
            with torch.set_grad_enabled(True):
                preds = model(X)
                _, y_pred = torch.max(preds, axis=1)
                loss = loss_fun(preds, y)

                # backpropagation (computes derivates)
                loss.backward()

                # optimizer step (updates parameters)
                optimizer.step()

            if batch % 1 == 0:
                print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{dl_size:>5f}]")

            running_loss += loss.item() * X.size(0)
            running_acc += torch.sum(y_pred == y)
        
        epoch_loss = running_loss / dl_size
        epoch_acc = running_acc / dl_size
        print('\n {} Loss: {:.4f} Acc: {:.4f}'.format(
            "Train", epoch_loss, epoch_acc))

    def val(
        self,
        model: Module,
        data_loader: DataLoader,
        loss_fun: Callable,
        device: str,
        models_folder: str,
        epoch: str,
        prev_loss: torch.tensor,
    ):
        self._val_step(
            model, data_loader, loss_fun, device, models_folder, epoch, prev_loss, self.test_dataloder_size
        )

    @staticmethod
    def _val_step(
        model: Module,
        data_loader: DataLoader,
        loss_fun: Callable,
        device: str,
        models_folder: str,
        epoch: str,
        prev_loss: torch.tensor,
        dl_size: int
    ):
        model.eval()
        
        running_loss, running_corrects = 0, 0
        
        for batch, (X, y) in enumerate(data_loader):
            y = y.to(device)
            X = X.float().to(device)

            with torch.set_grad_enabled(False):
                preds = model(X)
                _, y_preds = torch.max(preds, axis=1)
                loss = loss_fun(preds, y)
            
            running_loss += loss.item() * X.size(0) # computes sum of losses
            running_corrects += torch.sum(y_preds == y)

        epoch_loss = running_loss / dl_size # computes loss mean
        epoch_acc = running_corrects / dl_size # computes accuracy mean
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            "Validation", epoch_loss, epoch_acc))

        if epoch_loss < prev_loss:
            torch.save(
                model,
                f"{str(models_folder)}/{model.__module__}_epoch_{epoch}_metric_{epoch_acc}.pt",
            )

        return epoch_loss
