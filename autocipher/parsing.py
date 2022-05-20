import os
from datetime import datetime
from argparse import ArgumentParser, Namespace


def add_train_args(parser: ArgumentParser):
    parser.add_argument('--run_id',
                        type=str,
                        default='4_nt_enc_resdec',
                        help='Run ID')
    parser.add_argument('--device',
                        type=str,
                        choices=['cuda', 'cpu'],
                        default='cuda',
                        help='Device for training (default: cuda)')
    parser.add_argument('--data_path',
                        type=str,
                        default='./_assets/data/catsdogs',
                        help='Path to data')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./.assets/checkpoints',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--logs_dir',
                        type=str,
                        default='./.assets/logs',
                        help='Directory where Tensorboard logs will be saved')

    parser.add_argument('--seed',
                        type=int,
                        default=12,
                        help='Random seed to use when splitting data into train/val sets (default: 12)')
    parser.add_argument('--split_sizes',
                        type=float,
                        nargs=2,
                        default=[0.8, 0.2],
                        help='Split proportions for train/validation sets')
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='Number of workers for data loading (default: 4)',
                        )

    # Training arguments
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run (default: 100)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Initial learning rate (default: 1e-2)')



def modify_train_args(args: Namespace):
    if args.logs_dir is not None:
        timestamp = datetime.now().strftime('%y%m%d-%H%M%S%f')
        log_path = '{}_{}'.format(args.run_id, timestamp)
        args.logs_dir = os.path.join(args.logs_dir, log_path)

        if os.path.exists(args.logs_dir):
            num_ctr = 0
            while (os.path.exists(f'{args.logs_dir}_{num_ctr}')):
                num_ctr += 1
            args.logs_dir = f'{args.logs_dir}_{num_ctr}'

        os.makedirs(args.logs_dir)

    if args.save_dir is not None:
        timestamp = datetime.now().strftime('%y%m%d-%H%M%S%f')
        log_path = '{}_{}'.format(args.run_id, timestamp)
        args.save_dir = os.path.join(args.save_dir, log_path)

        if os.path.exists(args.save_dir):
            num_ctr = 0
            while (os.path.exists(f'{args.save_dir}_{num_ctr}')):
                num_ctr += 1
            args.save_dir = f'{args.save_dir}_{num_ctr}'

        os.makedirs(args.save_dir)


def parse_train_args() -> Namespace:
    parser = ArgumentParser()
    add_train_args(parser)
    temp_args, unk_args = parser.parse_known_args()
    args = parser.parse_args()
    modify_train_args(args)
    return args
