from typing import Callable
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm



def train(
        epoch: int,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: _LRScheduler or None,
        loss_func: Callable,
        writer: SummaryWriter = None):
    model.train()

    loss_sum  = 0

    for idx, (images, targets) in enumerate(tqdm(train_loader)):
        if next(model.parameters()).is_cuda:
            images, targets = images.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, targets)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_sum += loss.item()

        global_step = epoch * len(train_loader) + idx

        if writer is not None:
            writer.add_scalar('train/iter_loss', loss.item(),
                              global_step=global_step)

        if idx % 100 == 99:
            writer.add_images("train/iter_output", nn.Sigmoid()(outputs[:8]), global_step=global_step)
            writer.add_images("train/iter_target", targets[:8], global_step=global_step)

    if writer is not None:
        loss_avg = loss_sum / len(train_loader)

        print(f'Loss = {loss_avg:.4e}')

        writer.add_scalar('train/loss', loss_avg, global_step=epoch)
        writer.add_images("train/output", nn.Sigmoid()(outputs[:8]), global_step=epoch)
        writer.add_images("train/target", targets[:8], global_step=epoch)


    return loss_sum
