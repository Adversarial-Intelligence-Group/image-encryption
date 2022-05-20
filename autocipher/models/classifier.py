from sklearn import feature_extraction
import torch
from torchvision import models
import torch.nn as nn



class CLFNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        num_classes = 2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.15),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.15),
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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
