
import os
import random
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from autocipher.data import get_dataloaders
from autocipher.models import Autoencoder
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms as tf

from autocipher.models.cipher import Cipher
from autocipher.models.classifier import get_resnet


def plot_ae_data(args: Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        'cpu' if not torch.cuda.is_available() else args.device)

    train_loader, val_loader, test_loader = get_dataloaders(args, False)

    device = torch.device( 'cpu')

    model = Autoencoder()
    model.to(device)
    model.eval()

    # checkpoint_path = './.assets/checkpoints/ex_1_220515-222359423355/29/checkpoint.pth'
    checkpoint_path = './.assets/checkpoints/ex_5_220516-020959986245/99/checkpoint.pth'
    data_dir = './_assets/plots/'

    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])


    clf = get_resnet()
    clf_path = './.assets/checkpoints/ex_1_clf_220515-235814105533/9/checkpoint.pth'
    state = torch.load(clf_path)
    clf.load_state_dict(state['model'])
    fcclf = clf.fc
    clf.fc = nn.Identity()

    cipher = Cipher()
    cipher = cipher.to(device)

    images = next(iter(train_loader))[0]
    img_size = 224
    for i in range(10):
        images = images.to(device)

        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)

        orig = images[i]
        # latent_img = cipher(model.encoder(images))
        latent_img = cipher(clf(images))
        # decoded_img = model.decoder(latent_img)[i]
        decoded_img = orig
        lbl = fcclf(latent_img)[i]
        latent_img = latent_img[i]

        print(lbl)

        # ORIGINAL IMAGE
        img = tf.ToPILImage()(orig)
        plt.title('Original')
        plt.imshow(img)

        # LATENT IMAGE
        mx = latent_img.max()
        mn = latent_img[0].min()
        latent_flat = ((latent_img - mn) * 255/(mx - mn)).flatten()
        # 32*28*28
        # img = Image.fromarray( latent_flat[:24964].detach().cpu().numpy().astype('uint8').reshape((158, 158)), mode='L') 
        # img = Image.fromarray( latent_flat[:6241].detach().cpu().numpy().astype('uint8').reshape((79, 79)), mode='L') 
        img = Image.fromarray( latent_flat[:484].detach().cpu().numpy().astype('uint8').reshape((22, 22)), mode='L') 
        plt.subplot(1,3,2)
        plt.title('Latent')
        plt.xlim((-10, 30))
        plt.ylim((-10, 30))
        plt.axis('off')
        plt.imshow(img)

        # RECONSTRUCTED IMAGE
        img = tf.ToPILImage()(nn.Sigmoid()(decoded_img))
        plt.subplot(1,3,3)
        plt.title('Reconstructed')
        plt.imshow(img)

        plt.savefig(f'{data_dir}{i}_{lbl}.png')        
        plt.show()

if __name__ == "__main__":
    from autocipher.parsing import parse_train_args
    args = parse_train_args()
    plot_ae_data(args)
