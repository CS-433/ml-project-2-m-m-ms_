import argparse
import json
import torch
import os
import warnings
import random
import numpy as np
from trainer import Trainer

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

def main(args):
    """Main function to start the training."""

    config = json.load(open('config.json'))
    trainer_inst = Trainer(config, args.device)
    
    exp_dir = os.path.join(config['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer_inst.train()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
    args = parser.parse_args()

    main(args)