from turtle import forward
import torch
from torch import nn
from autocipher.models.cipher import Cipher

from autocipher.models.classifier import Resnet18, get_resnet

class Encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # nn.Dropout(0.15),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # nn.MaxPool2d(2, stride=2),
            # nn.Dropout(0.15),
            nn.Conv2d(128, 8, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(32),
            nn.ConvTranspose2d(8, 128, 1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(32, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, stride=1, padding=1),
        )


    def forward(self, x):
        return self.decoder(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # nn.Dropout(0.15),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # nn.MaxPool2d(2, stride=2),
            # nn.Dropout(0.15),
            nn.Conv2d(64, 32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(32),
            nn.ConvTranspose2d(32, 128, 1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(32, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, stride=1, padding=1),
        )


    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder1()
        # self.encoder = Resnet18()
        # self.cipher = Cipher()
        self.decoder = Decoder1()

    def forward(self, x):
        features = self.encoder(x)          
        out = self.decoder(features)
        return out
        




        # self.res_decoder = nn.Sequential(
        #     # nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
        #     # nn.InstanceNorm2d(32),
        #     nn.ConvTranspose2d(512, 256, 1, stride=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2.0),
        #     nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2.0),
        #     nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2.0),
        #     nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2.0),
        #     nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2.0),
        #     nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(8, 3, 3, stride=1, padding=1),
        # )