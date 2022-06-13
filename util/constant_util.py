"""Global constant definitions."""

import pathlib


def _get_root():
    """Retrieve the root directory of the project.

    Returns:
        A `PosixPath` object indicates the root directory.
    """
    return pathlib.Path.cwd()


# global constant for root directory
ROOT_DIR = _get_root()

# global constant for log dir
LOG_DIR = ROOT_DIR / 'snapshot'

# global constant for maximum training steps if user does not specify
MAXIMUM_TRAIN_STEPS = 2000000

# global constant for frequency of logging training info
LOG_TRAIN_INFO_STEPS = 1000

# global constant for frequency of saving checkpoint
KEEP_CKPT_STEPS = 10000

# global constant for frequency of validation
VALID_STEPS = 100000
