"""Define the utility functions for general purpose."""

from contextlib import contextmanager
import shutil

from util import logger


def is_subclass(derived, parent):
    """Check if one class is derived from another one.

    It also unwraps if class is wrapped by singleton.

    Args:
        derived: An object represents the derived child class.
        parent: An object represents the parent class.

    Returns:
        A bool indicates if `derived` is derived from `parent`.
    """
    # Unwrap class if it is wrapped (eg. wrapped by singleton)
    if hasattr(derived, '__wrapped__'):
        derived = derived.__wrapped__

    if hasattr(derived, '__bases__'):
        for base in derived.__bases__:
            # Check if base is `parent` class
            if id(base) == id(parent):
                return True

            # Recursive parent class
            if is_subclass(base, parent):
                return True

    return False


def copy_file(src_path, dest_dir):
    """Copy file from source path to destination directory.

    Args:
        src_path: A string indicates the source file path.
        dest_dir: A string indicates the destination file directory.

    Raises:
        FileNotFoundError: If the `src_path` or `dest_dir` is not existing.
    """
    try:
        shutil.copy(src_path, dest_dir)
    except FileNotFoundError as e:
        logger.exception(
            f'Source file {src_path} or destination file directory {dest_dir} is not existing.'
        )
        raise e
    except shutil.SameFileError:
        logger.warning(f'{src_path} and {dest_dir} are the same file.')
        pass


@contextmanager
def check_config(cfg):
    """Check if all the configuration arguments are correctly assigned and popped out.

    Args:
        cfg: A dict indicates the configuration.

    Raises:
        ValueError: If the configuration is not correctly assigned.
    """
    yield cfg
    # check if the config exist
    logger.check(not cfg, f'Some keys in the configuration are not supported: {cfg.keys()}')
