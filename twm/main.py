import argparse
import random
from copy import deepcopy

import numpy as np
import torch
import wandb

from config import CONFIGS
from trainer import Trainer


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--game', type=str)
        parser.add_argument('--config', type=str, default='default')
        parser.add_argument('--seed', type=int, default=None)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--buffer_device', type=str, default=None)
        parser.add_argument('--cpu_p', type=float, default=0.5)
        parser.add_argument('--wandb', type=str, default='disabled')
        parser.add_argument('--project', type=str, default=None)
        parser.add_argument('--group', type=str, default=None)
        parser.add_argument('--save', action='store_true', default=False)
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**args)

    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # https://pytorch.org/docs/stable/notes/randomness.html
        # slows down performance
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

    # improves performance, but is non-deterministic
    torch.backends.cudnn.benchmark = True

    if __debug__:
        print('Running in debug mode, consider using the -O python flag to improve performance')

    # enable wandb service (experimental, https://github.com/wandb/client/blob/master/docs/dev/wandb-service-user.md)
    # this hopefully fixes issues with multiprocessing
    wandb.require(experiment='service')

    buffer_device = args.buffer_device if args.buffer_device is not None else args.device

    config = deepcopy(CONFIGS[args.config])
    config.update({
        'game': args.game, 'seed': args.seed, 'model_device': args.device, 'buffer_device': buffer_device,
        'cpu_p': args.cpu_p, 'save': args.save
    })

    wandb.init(config=config, project=args.project, group=args.group, mode=args.wandb)
    config = dict(wandb.config)

    trainer = Trainer(config)
    trainer.print_stats()
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
