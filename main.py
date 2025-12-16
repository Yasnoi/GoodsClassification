import os
import argparse

from utils.config_loader import config_loader
from models.train import TrainNet


def main():
    parser = argparse.ArgumentParser(description='Goods Classification')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode to run: train or test (default: train)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file (default: configs/config.yaml)')
    args = parser.parse_args()
    config = config_loader(args.config)

    if config is None:
        return

    if args.mode == 'train':
        TrainNet(config).train()
    elif args.mode == 'test':
        pass


if __name__ == '__main__':
    main()
