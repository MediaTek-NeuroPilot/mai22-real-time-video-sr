"""Build the dataset."""

from dataset.reds import REDS
from util import logger


def build_dataset(dataset_config):
    """Build a dataset and make `DataLoader`.

     A sample `dataset_config` may look like the following
        dataset_config = {
            'dataloader_settings': {
                'train': {
                    'batch_size': 16,
                    'drop_remainder': True,
                    'shuffle': True
                },
                'val': {
                    'batch_size': 1
                }
            }
        }

    Args:
        dataset_config: A `dict` contains information to create a dataset.

    Returns:
        A dataset `dict` contains dataloader for different split.
    
    Raises:
        ValueError: If `split` is not supported in the dataset.
    """
    dataloader = {}

    dataloader_settings = dataset_config.pop('dataloader_settings')
    # create datasets and dataloaders with different splits.
    for split, dataloader_setting in dataloader_settings.items():
        # check the given split is valid or not.
        logger.check(
            split.lower() in [x.lower() for x in REDS.SPLITS],
            f'Unexpected dataset split `{split}`'
        )

        # build dataset
        dataset = REDS(dataset_config, split=split.lower())
        dataset = dataset.build(**dataloader_setting)

        dataloader[split.lower()] = dataset

    return dataloader
