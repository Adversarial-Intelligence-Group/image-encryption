from typing import List, Union
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter


def clip_data(parameters: torch.Tensor, min_clip_value: float, max_clip_value: float) -> torch.Tensor:
    device = parameters.data.device

    min_clip_value = float(min_clip_value)
    max_clip_value = float(max_clip_value)

    if float(parameters.item()) > max_clip_value or float(parameters.item()) < min_clip_value:
        print(parameters.item())  # TODO del
        return parameters.data.clamp(
            min=min_clip_value, max=max_clip_value).to(device)
    return parameters


def save_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, path)


def load_checkpoint(path: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler,
                    cuda: bool = True):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    if cuda:
        print('Moving model to cuda')
        model = model.cuda()

    return model
