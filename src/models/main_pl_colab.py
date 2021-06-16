import argparse
import sys
from pytorch_lightning import callbacks
import torch
from torch import nn, optim
from src.data.make_dataset import mnist
from torchvision import datasets, transforms
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import math

data_dir = 'data/'


class MNIST(datasets.MNIST):
    @property
    def raw_folder(self) -> str:
        return self.root + "/raw/"

    @property
    def processed_folder(self) -> str:
        return self.root + "/processed/"


class MNIST_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        log_ps = self(images)
        loss = self.criterion(log_ps, labels)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.003)
        return optimizer


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(data_dir, download=True, train=True)
        MNIST(data_dir, download=True, train=False)

    def setup(self, stage: str = None):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))])

        self.trainset = MNIST(data_dir,
                              download=True,
                              train=True,
                              transform=transform)
        self.testset = MNIST(data_dir,
                             download=True,
                             train=False,
                             transform=transform)

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(self.trainset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)
        return trainloader

    def test_dataloader(self):
        testsetloader = torch.utils.data.DataLoader(self.testset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True)
        return testsetloader


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


if __name__ == '__main__':
    data = MNISTDataModule()
    model = MNIST_model()
    wandb_logger = WandbLogger()
    callbacks = [PrintCallback()]
    trainer = Trainer(logger=wandb_logger,
                      max_epochs=1,
                      limit_train_batches=0.2,
                      accelerator='ddp',
                      precision=16,
                      gpus=1,
                      callbacks=callbacks)
    trainer.fit(model, data)