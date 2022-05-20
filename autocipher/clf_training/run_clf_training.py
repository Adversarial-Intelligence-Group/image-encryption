import os
from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
from autocipher.models.cipher import Cipher

from autocipher.models.classifier import CLFNet, get_resnet

from .train import train
from .validate import validate
from autocipher.data import get_dataloaders
from .utils import load_checkpoint, save_checkpoint
import random
from autocipher.models import Autoencoder


def run_clf_training(args: Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        'cpu' if not torch.cuda.is_available() else args.device)

    writer = SummaryWriter(args.logs_dir)

    train_loader, val_loader, test_loader = get_dataloaders(args)


    model = CLFNet()
    model.to(device)
    
    ae_model = Autoencoder()
    ae_model.to(device)
    ae_model.eval()
    ae_checkpoint_path = './.assets/checkpoints/ex_5_220516-020959986245/99/checkpoint.pth'
    # ae_checkpoint_path = './.assets/checkpoints/ex_1_220515-222359423355/29/checkpoint.pth'
    state = torch.load(ae_checkpoint_path)
    ae_model.load_state_dict(state['model'])

    cipher = Cipher()
    cipher = cipher.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # FIXME gamma
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)

    args.checkpoint_path = './.assets/checkpoints/ex_1_clf_220515-235814105533/9/checkpoint.pth'
    start_epoch = 0
    if args.checkpoint_path is not None:
        start_epoch = 0 # FIXME
        print(f'Loading model from {args.checkpoint_path}')
        model = load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            scheduler,
            args.device == 'cuda'
        )


    val_loss, val_accuracy = 0, 0
    train_loss, train_accuracy = 0, 0

    best_val_loss = 100
    best_counter = 0

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_losses, train_accs = train(epoch, model, ae_model, cipher, train_loader, optimizer,
                                         None, loss_func, writer)
        val_losses, val_accs = validate(
            epoch, model, ae_model, cipher,  val_loader, loss_func, writer)
        scheduler.step()

        writer.flush()

        train_loss += train_losses
        train_accuracy += train_accs
        val_loss += val_losses
        val_accuracy += val_accs

        if args.save_dir is not None:
            checkpoint_path = os.path.join(
                args.save_dir, str(epoch))
            os.makedirs(checkpoint_path, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
            save_checkpoint(checkpoint_path, model, optimizer, scheduler)
        
        if best_val_loss >= (val_losses/len(val_loader)):
            best_val_loss = (val_losses/len(val_loader))
            best_counter = 0
        else:
            best_counter += 1
        
        if best_counter == 5:
            break

    # results_root = Path(args.save_dir).parent
    df = pd.DataFrame(data={
        'Device': [args.device],
        'Epochs': [args.epochs],
        'Batch size': [args.batch_size],
        'Learning rate': [args.lr],
        'Accuracy train': [np.mean(train_accuracy)],
        'Accuracy validation': [np.mean(val_accuracy)],
        'Loss train': [np.mean(train_loss)],
        'Loss validation': [np.mean(val_loss)],
    })


    # save_path = os.path.join(results_root, args.run_id)
    # os.makedirs(save_path, exist_ok=True)
    save_path = args.save_dir
    df.to_json(os.path.join(save_path, 'final.json'), orient='records')
