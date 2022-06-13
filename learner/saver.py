"""Define the saver to save and load checkpoints."""

import pathlib

import tensorflow as tf

from util import common_util, logger


class Saver:
    """Implement the saver to save and load checkpoints.

    The checkpoints must include state of model, state of optimizer, step and epoch,
        and optionally contain state of learning rate scheduler.

    Attributes:
        learner: A `learner` object. See `core.learner.learner_base`.
        is_train: A `bool` indicates whether the current process is training or testing.
        log_dir: A `str` represents the directory where to save checkpoints.
        restore_ckpt: A `str` represents the path to checkpoint where would be restored from.
    """

    def __init__(self, saver_config, learner, is_train, log_dir=None):
        """Initialize the saver.

        A sample `saver_config` may look like the following
            saver_config = {
                'restore_ckpt': 'path/to/checkpoint'
            }
            restore_ckpt: Please refer to Attributes.

        Args:
            saver_config: A `dict` defines all parameters to build the saver.
            learner: Please refer to Attributes.
            is_train: Please refer to Attributes.
            log_dir: Please refer to Attributes. Defaults to None.

        Raises:
            KeyError: If any required argument is not provided.
            ValueError: If some keys in the configuration are not supported.
        """
        self.learner = learner
        self.is_train = is_train
        self.log_dir = pathlib.Path(log_dir) if log_dir else None

        with common_util.check_config(saver_config):
            self.restore_ckpt = saver_config.pop('restore_ckpt', None)

        if is_train:
            self.ckpt = tf.train.Checkpoint(
                model=self.learner.model, optimizer=self.learner.optimizer
            )
        else:
            self.ckpt = tf.train.Checkpoint(model=self.learner.model)

    def save_checkpoint(self):
        """Save checkpoint to `log_dir`.

        Raises:
            ValueError: If `log_dir` is None.
        """
        logger.check(self.log_dir is not None, '`log_dir` should be initialized')

        # save checkpoint
        self.ckpt.save(self.log_dir / 'ckpt')

    def load_checkpoint(self):
        """Load checkpoint from `restore_ckpt`.

        Raises:
            ValueError: If `restore_ckpt` is None.
        """
        logger.check(self.restore_ckpt is not None, '`restore_ckpt` should be initialized')

        # load checkpoint
        if self.is_train:
            status = self.ckpt.restore(self.restore_ckpt)
            status.assert_existing_objects_matched()
            self.learner.step = self.learner.optimizer.iterations.numpy()
        else:
            status = self.ckpt.restore(self.restore_ckpt).expect_partial()
            status.assert_existing_objects_matched()
