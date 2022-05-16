from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm



def validate(epoch: int,
             model: nn.Module,
             val_loader: DataLoader,
             loss_func: Callable[[Tensor, Tensor], Tensor],
             writer: SummaryWriter = None):
    model.eval()

    loss_sum = 0

    for idx, (images, targets) in enumerate(tqdm(val_loader)):
        if next(model.parameters()).is_cuda:
            images, targets = images.cuda(), targets.cuda()

        with torch.no_grad():
            if writer is not None and (epoch == 0 and idx == 0):
                writer.add_graph(model, images)

            outputs = model(images)
            loss = loss_func(outputs, targets)
            loss_sum += loss.item()

        global_step = epoch * len(val_loader) + idx

        if writer is not None:
            writer.add_scalar(
                'val/iter_loss', loss.item(), global_step=global_step)

    if writer is not None:
        loss_avg = loss_sum / len(val_loader)

        print(f'Loss = {loss_avg:.4e}')

        writer.add_scalar('val/loss', loss_avg, global_step=epoch)
        writer.add_images("val/output", nn.Sigmoid()(outputs[:8]), global_step=epoch)
        writer.add_images("val/target", targets[:8])


    return loss_sum
