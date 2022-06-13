"""Generic main function for VSR experiments."""

import argparse
import pathlib

import tensorflow as tf
import yaml

from dataset import dataset_builder
from learner.learner import StandardLearner
from util import common_util, constant_util, plugin


def _parse_argument():
    """Return arguments for VSR."""
    parser = argparse.ArgumentParser(description='Codebase for MAI 2022 VSR.')
    parser.add_argument(
        '--process',
        help='Process type.',
        type=str,
        default='train',
        choices=['train', 'test'],
        required=True
    )
    parser.add_argument(
        '--config_path',
        help='Path of yaml config file of the application.',
        type=str,
        default=None,
        required=True
    )

    args = parser.parse_args()

    return args


def main(args):
    """Run main function for vision quality experiments.

    Args:
        args: A `dict` contain augments 'process' and 'config_path'.

    Raises:
        ValueError:
            1. If test dataset can't be prepared.
            2. If process type is not 'train' or 'test'.
    """
    # prepare configurations
    with open(args.config_path, 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    log_dir = config.pop('log_dir', constant_util.LOG_DIR)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    common_util.copy_file(args.config_path, log_dir)

    # prepare dataset
    dataset = dataset_builder.build_dataset(config['dataset'])

    # prepare model
    model_builder = plugin.plugin_from_file(
        config['model']['path'], config['model']['name'], tf.keras.Model
    )
    common_util.copy_file(config['model']['path'], log_dir)
    model = model_builder()

    # prepare learner
    learner = StandardLearner(config['learner'], model, dataset, log_dir)

    if args.process == 'train':
        learner.train()
    elif args.process == 'test':
        learner.test()
    else:
        raise ValueError(f'Wrong process type {args.process}')


if __name__ == '__main__':
    arguments = _parse_argument()
    main(arguments)