from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from autocipher.models.autoencoder import Autoencoder
from autocipher.models.cipher import Cipher

from .utils import accuracy


def validate(epoch: int,
             model: nn.Module,
             ae_model: Autoencoder,
             cipher: Cipher,
             val_loader: DataLoader,
             loss_func: Callable[[Tensor, Tensor], Tensor],
             writer: SummaryWriter = None):
    model.eval()

    loss_sum, accs_sum = 0, 0

    for idx, (images, targets) in enumerate(tqdm(val_loader)):
        if next(model.parameters()).is_cuda:
            images, targets = images.cuda(), targets.cuda()

        with torch.no_grad():
            if writer is not None and (epoch == 0 and idx == 0):
                writer.add_graph(model, images)

            images = nn.Sigmoid()(ae_model.decoder(cipher(ae_model.encoder(images))))
            # images = ae_model(images)
            outputs = model(images)
            acc = accuracy(outputs, targets)
            loss = loss_func(outputs, targets)
            loss_sum += loss.item()
            if isinstance(acc, torch.Tensor):
                accs_sum += acc.item()
            else:
                accs_sum += acc

        global_step = epoch * len(val_loader) + idx

        if writer is not None:
            writer.add_scalar(
                'val/iter_loss', loss.item(), global_step=global_step)

    if writer is not None:
        loss_avg = loss_sum / len(val_loader)
        accs_avg = accs_sum / len(val_loader)

        print(f'Loss = {loss_avg:.4e}, Accuracy = {accs_avg:.4e}')

        writer.add_scalar('val/loss', loss_avg, global_step=epoch)
        writer.add_scalar('val/accuracy', accs_avg, global_step=epoch)

    return loss_sum, accs_sum
