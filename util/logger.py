"""Define logging utilities."""

import logging
import os
import sys
import threading

from util import constant_util

# redirect LOG level for ease of use
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
_LEVEL_BOOK = {
    'debug': DEBUG, 'info': INFO, 'warning': WARNING, 'error': ERROR, 'critical': CRITICAL
}


def _interpret_level(key):
    """Interpret debug level by given key string.

    Args:
        key: A string indicates the value of the environment variable key.

    Returns:
        An int indicates the log level.
    """
    if key is not None:
        if key in _LEVEL_BOOK.keys():
            return _LEVEL_BOOK[key]
    return INFO


# logger name, also prefix to each log line.
_LOGGER_NAME = 'MAI VSR'  # logger name, also the log prefix

# log level preference, default is INFO
# mai_vsr_LOG_LEVEL have lower priority than mai_vsr_NDEBUG
_LOG_LEVEL = _interpret_level(os.getenv('VA_LOG_LEVEL', None)) if __debug__ else\
    max(_interpret_level(os.getenv('VA_LOG_LEVEL', None)), ERROR)

# enable to log streaming or file
# at least one of streaming and file should be enabled
_LOG_STREAM = os.getenv('VA_LOG_EN_STREAM', 'True') not in ('False', 'false', '0')
_LOG_FILE = os.getenv('VA_LOG_EN_FILE', 'True') not in ('False', 'false', '0')
_LOG_FILE_PATH = constant_util.ROOT_DIR / os.getenv('VA_LOG_FILE', 'mai_vsr_logger.log')
_LOG_FORMAT = '%(asctime)s [%(name)s:%(levelname)s] %(filename)s:%(lineno)d: %(message)s'
assert _LOG_STREAM is True or _LOG_FILE is True, 'None of LOG_STREAM and LOG_FILE enabled'

# global logger
_logger = None
_lock = threading.Lock()  # lock for thread safety logger creation


def _find_caller(stack_info=False):
    """Track back system caller stack to find caller file, function and lineno.

    Args:
        stack_info: A bool indicates how the stack information be returned. The stack inforation
            is returned as None unless `stack_info` is True. Default is False.

    Returns:
        4-element tuple, which represent caller's filename, lineno, code name, stack info.
            Reference to `logging` for full spec.

    Raises:
        Exceptions raised during tracing caller stack.
    """
    try:
        # logger_f = execute frame of this file, this function.
        logger_f = sys._getframe(3)  # pylint: disable=protected-access

        # track back until caller frame is found
        # this is needed because we may redirect function call within this logger.
        caller_f = logger_f
        while caller_f.f_code.co_filename == logger_f.f_code.co_filename:
            caller_f = caller_f.f_back

        # stack info if required
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())

        # return tuple information according to logging's findcaller.
        return (caller_f.f_code.co_filename, caller_f.f_lineno, caller_f.f_code.co_name, sinfo)

    except:
        print('Unexpected runtime caller stack.')
        raise


def _get_logger():
    """Get logger, or create one if logger is not exist.

    Intended to create mai_vsr logger instance, or return created logger directly. The logger is
    created by `logging`, python built-in logging utilities.
    """
    global _logger  # pylint: disable=global-statement

    # Return logger if already exist
    if _logger:
        return _logger

    _lock.acquire()
    try:
        # return logger if already exist, in lock section because it may created by other thread.
        if _logger:
            return _logger

        # create logger with name 'mai_vsr'
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(_LOG_LEVEL)
        logger.findCaller = _find_caller

        # log format, e.g. [mai_vsr:INFO] install.py:7: It is log message
        formatter = logging.Formatter(_LOG_FORMAT)
        if _LOG_STREAM:
            # setup stream handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if _LOG_FILE:
            # setup file handler
            fh = logging.FileHandler(_LOG_FILE_PATH.as_posix())
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        _logger = logger
        return _logger

    finally:
        _lock.release()


def _color_string(msg, color=None):
    """Colour the given string.

    Args:
        msg: A string indicate the string being coloured.
        color: A string indicates the identifier to the color book. Default to None.

    Returns:
        Coloured string if color is given, original string if color is None.

    Raises:
        ValueError: If invalid color identifier occurs.
    """
    color_book = {
        'red': '\33[91m',
        'green': '\33[92m',
        'yellow': '\33[93m',
        'blue': '\33[94m',
        'white': '\33[0m',
    }
    if color is not None:
        if color not in color_book.keys():
            raise ValueError('Unknown color')
        return color_book[color] + msg + color_book['white']
    return msg


def set_level(level):
    """Interface to change logging level.

    Args:
        level: An int indicates the logging level, must be one of DEBUG, INFO, WARNING, ERROR,
            CRITICAL.

    Raises:
        ValueError: If the level is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    if level not in [DEBUG, INFO, WARNING, ERROR, CRITICAL]:
        raise ValueError('Unknown level')
    _get_logger().setLevel(level)


def set_logfile_path(file_path):
    """Set the logfile path for the file handler.

    Remove the original file handler, then set a new one.

    Args:
        file_path: A string or `pathlib.Path` object. The path of the log file.
    """
    global _logger  # pylint: disable=global-statement

    if not _logger:
        _get_logger()

    _lock.acquire()
    try:
        # remove the existed `FileHandler`.
        for handler in _logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
                handler.close()
                _logger.removeHandler(handler)

        # set the new file handler.
        fh = logging.FileHandler(file_path)
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        _logger.addHandler(fh)

    finally:
        _lock.release()


def log(level: int, msg: str, *args, **kwargs):
    """Log message in given log level."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().log(level, msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    """Log message in DEBUG level."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log message in INFO level."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log message in WARNING level."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log message in ERROR level."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().error(msg, *args, **kwargs)


def fatal(msg: str, *args, **kwargs):
    """Log message in FATAL(=CRITICAL) level."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().fatal(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log message in CRITICAL level."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """Log message with specific exception message."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().exception(msg, *args, **kwargs)


def vlog(msg: str, *args, **kwargs):
    """Log message with INFO level, and extra level by verbose."""
    msg = _color_string(msg, kwargs.pop('color', None))
    _get_logger().info(msg, *args, **kwargs)


def vlog_if(condition: bool, msg: str, *args, **kwargs):
    """Conditional log message with INFO level, and extra level by verbose."""
    if condition:
        msg = _color_string(msg, kwargs.pop('color', None))
        _get_logger().info(msg, *args, **kwargs)


def check(condition, msg, *args, **kwargs):
    """Check the condition, log and raise if the condition is not meet.

    Args:
        condition: A bool indicates if the assigned condition is satisfied or not. If False, the
            condition is not satisfied.
        msg: A string indicates the log message.

    Raises:
        ValueError: If the condition is not satisfied.
    """
    if not condition:
        msg = 'Check failed: ' + msg
        msg = _color_string(msg, kwargs.pop('color', 'yellow'))
        _get_logger().error(msg, *args, **kwargs)
        raise ValueError('Condition check failed.')


def dcheck(condition, msg, *args, **kwargs):
    """In debug version, check the condition, log and raise if the condition is not meet.

    Args:
        condition: A bool indicates if the assigned condition is satisfied or not. If False, the
            condition is not satisfied.
        msg: A string indicates the log message.

    Raises:
        ValueError: If the condition is not satisfied.
    """
    if __debug__:
        if not condition:
            msg = 'Check failed: ' + msg
            msg = _color_string(msg, kwargs.pop('color', 'yellow'))
            _get_logger().error(msg, *args, **kwargs)
            raise ValueError('Condition check failed.')


def check_type(value, expected_type, *args, **kwargs):
    """Check the value type, log and raise if the value type is not expected.

    Args:
        value: The value to be checked.
        expected_type: The expected type of the value. It could be a type or tuple of types.
        kwargs:
            msg: The specific logging message. If not specified, use the default logging message.
                Defaults to None.

    Raises:
        TypeError: If the type of value is not expected.
        TypeError: If the `expected_type` is not a type. Raised by `isinstance`.
    """
    if not isinstance(value, expected_type):
        msg = kwargs.pop('msg', None)
        msg = 'Check failed: ' + msg if msg else\
            f'Expected type {expected_type}, but received {value}.'
        msg = _color_string(msg, kwargs.pop('color', 'yellow'))
        _get_logger().error(msg, *args, **kwargs)
        raise TypeError('The type is not expected.')


def dcheck_type(value, expected_type, *args, **kwargs):
    """In debug version, check the value type, log and raise if the value type is not expected.

    Args:
        value: The value to be checked.
        expected_type: The expected type of the value. It could be a type or tuple of types.
        kwargs:
            msg: The specific logging message. If not specified, use the default logging message.
                Defaults to None.

    Raises:
        TypeError: If the type of value is not expected.
        TypeError: If the `expected_type` is not a type. Raised by `isinstance`.
    """
    if __debug__:
        if not isinstance(value, expected_type):
            msg = kwargs.pop('msg', None)
            msg = 'Check failed: ' + msg if msg else\
                f'Expected type {expected_type}, but received {value}.'
            msg = _color_string(msg, kwargs.pop('color', 'yellow'))
            _get_logger().error(msg, *args, **kwargs)
            raise TypeError('The type is not expected.')


def unreachable(*args, **kwargs):
    """Mark unreachable code, and raise runtime exception if program executed.

    Raises:
        RuntimeError: If the unreachable code is executed.
    """
    if __debug__:
        msg = kwargs.pop('msg', None)
        msg = 'Unreachable error ' + (msg if msg else '')
        msg = _color_string(msg, kwargs.pop('color', 'yellow'))
        _get_logger().error(msg, *args, **kwargs)
        raise RuntimeError('Execute unreachable code.')
