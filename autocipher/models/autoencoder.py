from turtle import forward
import torch
from torch import nn
from autocipher.models.cipher import Cipher

from autocipher.models.classifier import Resnet18, get_resnet
from autocipher.models.vgg import vgg11


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = Encoder()
        self.encoder = VGGEncoder()
        # self.encoder = Resnet18()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vgg11(True).features

    def forward(self, x):
        return self.encoder(x)
        

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


class ResnetDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.res_decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            # nn.InstanceNorm2d(32),
            nn.ConvTranspose2d(512, 256, 1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0),
            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.res_decoder(x)
