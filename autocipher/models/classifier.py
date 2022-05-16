from sklearn import feature_extraction
import torch
from torchvision import models
import torch.nn as nn


class Resnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = models.resnet18(True)

        feature_extracting = False
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

        self.layers = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.layers(x)

def get_resnet():
    model = models.resnet18(True)

    feature_extracting = True
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    
    # model.fc = nn.Identity()
    num_ftrs = model.fc.in_features
    num_classes = 2
    model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    return model
