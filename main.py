import argparse
import logging
import os
import shutil
import sys
import tensorboardX as tb

import numpy as np
import torch
import yaml

from configs import *
from runners import *


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='warning', help='Verbose level: info | debug | warning | critical')

    # Task to do
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('--fast_fid', action='store_true', help='Whether to evaluate the FID metric')
    parser.add_argument('--inception', action='store_true', help='Whether to evaluate the inception metric')
    parser.add_argument('--eval', action='store_true', help='')
    parser.add_argument('--stackedmnist', action='store_true', help='Whether to execute the StackedMNIST task')

    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--data_folder', default='/Datasets')
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--fid_folder', default='/scratch', help='where to store FID results')

    # Override parameters
    parser.add_argument('--consistent', action='store_true', help='If True, overwrite yml file and use consistent sampling')
    parser.add_argument('--nsigma', type=int, default=0, help='number of steps per sigmas (or multiplier on number of sigmas if consistent=true)')
    parser.add_argument('--step_lr', type=float, default=0, help='sampling.step_lr')
    parser.add_argument('--batch_size', type=int, default=0, help='sampling/fid batch size')
    parser.add_argument('--begin_ckpt', type=int, default=0, help='sampling.begin_ckpt')
    parser.add_argument('--end_ckpt', type=int, default=0, help='sampling.end_ckpt')
    parser.add_argument('--fid_num_samples', type=int, default=0, help='fast_fid.num_samples')
    parser.add_argument('--adam', action='store_true', help='If True, uses adam_beta instead of the ones in config')
    parser.add_argument('--adam_beta', nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument('--D_adam', action='store_true', help='If True, uses D_adam_beta with Discriminator instead of the ones in config')
    parser.add_argument('--D_adam_beta', nargs=2, type=float, default=(0.9, 0.999))
    parser.add_argument('--D_steps', type=int, default=0, help='Number of discriminator per score network steps')

    # Adversarial option
    parser.add_argument('--adversarial', action='store_true', help='Adversarial Denoising autoencoder')

    # Loss functions options
    parser.add_argument('--target', type=str, default='gaussian', help='dae: predict x and apply a sigmoid, gaussian:(x - x_tilde)/sigma')
    parser.add_argument('--loss', type=str, default='l2', help='Amongst "l1", "l2" and "l1_msssim"')

    # Model
    parser.add_argument('--std', action='store_true', help='divide score network by variance (so its always standardized)')
    parser.add_argument('--no_dilation', action='store_true',
                        help='If True, do not use dilation anywhere in architecture. This should reduces memory requirement and increase speed significantly.')

    # Find best hyperameters
    parser.add_argument('--compute_approximate_sigma_max', action='store_true',
                        help='sigma_max must be the maximum pairwise eucledian distance between points, we can get a really rough estimate by only looking at mini-batches pairwise distances. This will run for one epoch.')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    #### Specific to ComputeCanada Beluga server
    #args.data_folder = os.environ["SLURM_TMPDIR"] + "/" + args.data_folder
    #args.fid_folder = args.fid_folder + "/" + os.environ["USER"] + "/Output" + "/Extra"
    
    os.makedirs(args.fid_folder, exist_ok=True)

    config = get_config(args)

    args.image_folder_denoised = args.image_folder

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)
    args.level = getattr(logging, args.verbose.upper(), None)

    validate_args(args)

    if args.train:
        if not args.resume_training:
            ask_overwrite_folder(args.log_path, no_interactions=args.ni)
            ask_overwrite_folder(tb_path, no_interactions=args.ni)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        args.tb_logger = tb.SummaryWriter(log_dir=tb_path)

        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    elif not args.eval:
        path_dir = 'image_samples'

        os.makedirs(os.path.join(args.exp, path_dir), exist_ok=True)
        args.image_folder = os.path.join(args.exp, path_dir, args.image_folder)
        args.image_folder_denoised = os.path.join(args.exp, path_dir + '_denoised', args.image_folder_denoised)

        ask_overwrite_folder(args.image_folder, no_interactions=args.ni)
        ask_overwrite_folder(args.image_folder_denoised, no_interactions=args.ni)

    # setup logger
    logger = logging.getLogger()
    logger.setLevel(args.level)

    handlers = [logging.StreamHandler()]
    if args.train:
        handlers += [logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))]

    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    for h in handlers:
        h.setFormatter(formatter)
        logger.addHandler(h)

    # add device
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(args.device))  # TODO can i do that instead of args.device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config


def ask_overwrite_folder(folder, no_interactions, fatal=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif no_interactions:
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        response = input(f"Folder '{folder}' already exists. Overwrite? (Y/N)")
        if response.upper() == 'Y':
            shutil.rmtree(folder)
            os.makedirs(folder)
        elif fatal:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

def validate_args(args):
    assert args.target in ["gaussian", "dae"]
    assert isinstance(args.level, int)


def get_runner(args):
    assert args.train + args.sample + args.fast_fid + args.inception + args.eval + args.stackedmnist == 1
    if args.sample:
        return SampleRunner
    elif args.fast_fid:
        return FidRunner
    elif args.eval:
        return EvalRunner
    elif args.inception:
        return InceptionRunner
    elif args.stackedmnist:
        return StackedMNISTRunner
    else:
        return TrainRunner


def main():
    args, config = parse_args_and_config()

    print_config(config)
    get_runner(args)(args, config).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
