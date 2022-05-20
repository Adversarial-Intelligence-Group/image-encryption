import torch.nn as nn
import torch 
import os

# 32*28*28
class Cipher(nn.Module):
    def __init__(self) -> None:
        super().__init__()        
        in_size = 512*7*7#8*28*28
        self.layers = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(in_size, in_size),
            # nn.Tanh(),
            # nn.Linear(in_size, in_size),
            # nn.Tanh(),
            # nn.Linear(in_size, in_size),
            # nn.Tanh(),
            nn.Conv2d(512, 512, 1, 1),
            nn.Tanh(),
            # nn.ReLU(True),
            # nn.Linear(in_size//4, in_size),
            # nn.ReLU(True),
            # nn.Linear(in_size, in_size),
            # nn.ReLU(True),
            # nn.Unflatten(1, (512, 7, 7))
        )
        model_path = './_assets/cipher.pth'
        is_init = os.path.exists(model_path)
        if not is_init:
            self._initialize_weights()
            torch.save(self.state_dict(), model_path)
        else:
            self.load_state_dict(torch.load(model_path))

    def forward(self, x):
        return self.layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                # nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                # nn.init.normal_(m.bias, 0, 0.5)