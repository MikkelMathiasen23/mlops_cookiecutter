import argparse
import sys
import torch
from torch import nn, optim
from src.data.make_dataset import mnist
from torchvision import datasets, transforms
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import math
import torchdrift
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

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
    def __init__(self,
                 batch_size: int = 32,
                 additional_transform: str = False):
        super().__init__()
        self.batch_size = batch_size
        self.additional_transform = additional_transform
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

    def collate_fn(self, batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        if self.additional_transform:
            batch = (self.corruption_function(batch[0]), *batch[1:])
        return batch

    def prepare_data(self):
        MNIST(data_dir, download=True, train=True)
        MNIST(data_dir, download=True, train=False)

    def corruption_function(self, x: torch.Tensor):
        return torchdrift.data.functional.gaussian_blur(x, severity=2)

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(self.trainset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  collate_fn=self.collate_fn)
        return trainloader

    def test_dataloader(self):
        testsetloader = torch.utils.data.DataLoader(self.testset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    collate_fn=self.collate_fn)
        return testsetloader


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


def driftdetector():
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    return drift_detector


if __name__ == '__main__':
    drift = driftdetector()
    mapper = Isomap(n_components=2)

    #Load data:
    data = MNISTDataModule()
    ood_data = MNISTDataModule(additional_transform=True)

    model = MNIST_model()
    wandb_logger = WandbLogger()
    callbacks = [PrintCallback()]
    trainer = Trainer(max_epochs=1,
                      limit_train_batches=0.2,
                      accelerator='ddp',
                      gpus=None,
                      callbacks=callbacks)  #logger=wandb_logger,
    #trainer.fit(model, data)
    torchdrift.utils.fit(ood_data.train_dataloader(), model, drift)
    drift_model = nn.Sequential(model, drift)
    print('1')
    ###Visualize:
    inputs, _ = next(iter(ood_data.train_dataloader()))
    print('1')
    score = drift_model(inputs)
    print('1')
    p_val = drift_model.compute_p_value(inputs)
    print('1')
    base_embedded = mapper.fit_transform(drift_model.base_outputs)
    print('1')
    features_embedded = mapper.transform(inputs)
    print('1')
    plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
    plt.title(f'score {score:.2f} p-value {p_val:.2f}')
    plt.savefig('reports/figures/isomap.png')
