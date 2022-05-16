
from .utils import load_checkpoint, save_checkpoint
from argparse import Namespace

import torch
import torch.nn as nn
from autocipher.models import Autoencoder
import random
import os
import numpy as np
from autocipher.data import get_dataloaders


def save_ae_data(args: Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        'cpu' if not torch.cuda.is_available() else args.device)

    train_loader, val_loader, test_loader = get_dataloaders(args, False)

    ae_model = Autoencoder()
    ae_model.to(device)
    ae_model.eval()
    ae_checkpoint_path = './.assets/checkpoints/ex_1_220515-222359423355/29/checkpoint.pth'
    state = torch.load(ae_checkpoint_path)
    ae_model.load_state_dict(state['model'])