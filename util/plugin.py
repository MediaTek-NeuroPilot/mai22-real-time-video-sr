"""Define plugin utilities."""

import importlib.util
import pathlib

from util import logger
from util.common_util import is_subclass


def plugin_from_file(location, name, base):
    """Load plugin from source file.
    Args:
        location: A string represents the plugin source file.
        name: A string represents the class name of plugin and the class must exist in
            `location` file.
        base: A class indicates the expected class of plugin class.

    Return:
        A loaded plugin class.

    Raises:
        AttributeError: The given plugin class name is not found.
        ValueError: The plugin is not derived from `base`.
    """
    # dynamic load python module from location
    # it will throw FileNotFoundError, AttributeError if unexpected file
    spec = importlib.util.spec_from_file_location(pathlib.PurePath(location).stem, location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # find the requested plugin class
    try:
        plugin = getattr(module, name)
    except AttributeError:
        logger.error(f'Given plugin name {name} is not found.')
        raise

    if base:
        logger.check(is_subclass(plugin, base), 'Unexpected plugin base.')

    return plugin
