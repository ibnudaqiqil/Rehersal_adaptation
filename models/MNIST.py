import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST


class LitMNIST(LightningModule):
    def __init__(self, data_dir="./store/dataset", num_classes=10, hidden_size=64, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = num_classes
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
      
        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y,t = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([x['loss'] for x in outputs]))
        self.log('train_loss', loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y,t = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        #self.log("val_loss", loss, prog_bar=True)
        #self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack(outputs))
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', self.accuracy, on_epoch=True,
                 prog_bar=True)  # log scalar

    def test_step(self, batch, batch_idx):
        x, y, t = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("Test_loss", loss, prog_bar=True)
        self.log("Test_acc", self.accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

   
