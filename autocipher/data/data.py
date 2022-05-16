import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets
import os


class AEDataset(Dataset):
    def __init__(self, src_dir, lbls_dir, transform) -> None:
        super().__init__()
        self.samples = datasets.ImageFolder(src_dir,
                                     transform=transform)
        self.targets = datasets.ImageFolder(lbls_dir,
                                    transform=transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples.__getitem__(index)[0], self.targets.__getitem__(index)[0]


def get_ae_datasets(args):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train = AEDataset(os.path.join(args.data_path, 'train'),
                      os.path.join(args.data_path, 'train'),
                      transform=transform)
    test = AEDataset(os.path.join(args.data_path, 'val'),
                     os.path.join(args.data_path, 'val'),
                     transform=transform)

    return train, test


def get_clf_datasets(args):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225]),
    ])

    train = datasets.ImageFolder(os.path.join(args.data_path, 'train'),
                                 transform=transform)
    test = datasets.ImageFolder(os.path.join(args.data_path, 'val'),
                                transform=transforms.ToTensor())

    return train, test


def get_dataloaders(args, for_clf=True):
    if for_clf:
        dataset_train, dataset_test = get_clf_datasets(args)
    else:
        dataset_train, dataset_test = get_ae_datasets(args)

    train_size, val_size = args.split_sizes
    lengths = [int(len(dataset_train)*train_size),
               int(len(dataset_train)*val_size)]

    train_data, val_data = random_split(
        dataset_train, lengths, torch.Generator().manual_seed(args.seed))

    loader_args = dict(batch_size=args.batch_size,
                       num_workers=args.workers, drop_last=True)

    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    test_loader = DataLoader(dataset_test, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
