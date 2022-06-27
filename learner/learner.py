"""Define the learner."""

import pathlib
import random

import numpy as np
import tensorflow as tf

from learner.saver import Saver
from learner.metric import PSNR, SSIM
from util import common_util, constant_util, logger


class StandardLearner():
    """Implement the standard learner.

    Attributes:
        config: A `dict` contains the configuration of the learner.
        model: A list of `tf.keras.Model` objects which generate predictions.
        dataset: A dataset `dict` contains dataloader for different split.
        step: An `int` represents the current step. Initialize to 0.
        optimizer: A `tf.keras.optimizers` is used to optimize the model. Initialize to None.
        lr_scheduler: A `tf.keras.optimizers.schedules.LearningRateSchedule` is used to schedule
            the leaning rate. Initialize to None.
        metric_functions: A `dict` contains one or multiple functions which are used to
            metric the results. Initialize to {}.
        saver: A `Saver` is used to save checkpoints. Initialize to None.
        summary: A `TensorboardSummary` is used to save eventfiles. Initialize to None.
        log_dir: A `str` represents the directory which records experiments.
        steps: An `int` represents the number of train steps.
        log_train_info_steps: An `int` represents frequency of logging training information.
        keep_ckpt_steps: An `int` represents frequency of saving checkpoint.
        valid_steps: An `int` represents frequency  of validation.
    """

    def __init__(self, config, model, dataset, log_dir):
        """Initialize the learner and attributes.

        Args:
            config: Please refer to Attributes.
            model: Please refer to Attributes.
            dataset: Please refer to Attributes.
            log_dir: Please refer to Attributes.
        """
        super().__init__()
        with common_util.check_config(config['general']) as cfg:
            self.total_steps = cfg.pop('total_steps', constant_util.MAXIMUM_TRAIN_STEPS)
            self.log_train_info_steps = cfg.pop(
                'log_train_info_steps', constant_util.LOG_TRAIN_INFO_STEPS
            )
            self.keep_ckpt_steps = cfg.pop('keep_ckpt_steps', constant_util.KEEP_CKPT_STEPS)
            self.valid_steps = cfg.pop('valid_steps', constant_util.VALID_STEPS)

        self.config = config
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir

        self.step = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.metric_functions = {}
        self.saver = None
        self.summary = None

        # set random seed
        random.seed(2454)
        np.random.seed(2454)
        tf.random.set_seed(2454)

    def register_training(self):
        """Prepare for training."""
        # prepare learning rate scheduler for training
        lr_config = self.config['lr_scheduler'] if 'lr_scheduler' in self.config else {}
        module = getattr(tf.keras.optimizers.schedules, lr_config.pop('name'))
        self.lr_scheduler = module(**lr_config)

        # prepare optimizer for training
        opt_config = self.config['optimizer'] if 'optimizer' in self.config else {}
        module = getattr(tf.keras.optimizers, opt_config.pop('name'))
        self.optimizer = module(learning_rate=self.lr_scheduler, **opt_config)

        # prepare saver to save and load checkpoints
        saver_config = self.config['saver'] if 'saver' in self.config else {}
        self.saver = Saver(saver_config, self, is_train=True, log_dir=self.log_dir)

        # prepare metric functions
        self.metric_functions['psnr'] = PSNR(data_range=255)
        self.metric_functions['ssim'] = SSIM(data_range=255)

    def register_evaluation(self):
        """Prepare for evaluation."""
        # prepare saver to save and load checkpoints
        saver_config = self.config['saver'] if 'saver' in self.config else {}
        self.saver = Saver(saver_config, self, is_train=False, log_dir=self.log_dir)

        # prepare metric functions
        self.metric_functions['psnr'] = PSNR(data_range=255)
        self.metric_functions['ssim'] = SSIM(data_range=255)

    def loss_fn(self, pred_tensor, target_tensor):
        """Define the objective function and prepare loss for backward.

        Args:
            pred_tensor: A `torch.Tensor` represents the prediction.
            target_tensor: A `torch.Tensor` represents the target.
        """
        # l1 charbonnier loss
        epsilon = 1e-6
        diff = pred_tensor - target_tensor
        loss = tf.math.sqrt(diff * diff + epsilon)
        return tf.reduce_mean(loss)

    def log_metric(self, prefix=''):
        """Log the metric values."""
        metric_dict = {}
        with self.summary.as_default(step=self.step):
            for metric_name in self.metric_functions:
                value = self.metric_functions[metric_name].get_result().numpy()
                self.metric_functions[metric_name].reset()

                tf.summary.scalar(prefix + metric_name, value)
                metric_dict[metric_name] = value

        logger.info(f'Step: {self.step}, {prefix}Metric: {metric_dict}')
        self.summary.flush()

    @tf.function
    def train_step(self, data):
        """Define one training step.

        Args:
            data: A `tuple` contains input and target tensor.
        """
        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T

        l1_norm_loss = 0
        with tf.GradientTape() as tape:
            for i in range(recurrent_steps):
                if i == 0:
                    b, _, h, w, _ = input_tensors.shape.as_list()
                    input_tensor = tf.concat(
                        [input_tensors[:, 0, ...], input_tensors[:, 0, ...]], axis=-1
                    )
                    hidden_state = tf.zeros([b, h, w, self.model.base_channels])
                    pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=True)
                else:
                    input_tensor = tf.concat(
                        [input_tensors[:, i - 1, ...], input_tensors[:, i, ...]], axis=-1
                    )
                    pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=True)
                l1_norm_loss += self.loss_fn(pred_tensor, target_tensors[:, i, ...])
        # Calculate gradients and update.
        gradients = tape.gradient(l1_norm_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return pred_tensor, l1_norm_loss

    def test_step(self, data):
        """Define one testing step.

        Args:
            data: A `tuple` contains input and target tensor.
        """
        input_tensors, target_tensors = data
        recurrent_steps = target_tensors.shape[1]  # T

        pred_tensors = []
        for i in range(recurrent_steps):
            if i == 0:
                b, _, h, w, _ = input_tensors.shape.as_list()
                input_tensor = tf.concat(
                    [input_tensors[:, 0, ...], input_tensors[:, 0, ...]], axis=-1
                )
                hidden_state = tf.zeros([b, h, w, self.model.base_channels])
                pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=False)
            else:
                input_tensor = tf.concat(
                    [input_tensors[:, i - 1, ...], input_tensors[:, i, ...]], axis=-1
                )
                pred_tensor, hidden_state = self.model([input_tensor, hidden_state], training=False)

            for metric_name in self.metric_functions:
                self.metric_functions[metric_name].update(pred_tensor, target_tensors[:, i, ...])

            pred_tensors.append(pred_tensor)

        return pred_tensors

    def train(self):
        """Train the model."""
        self.register_training()
        self.summary = tf.summary.create_file_writer(self.log_dir)

        # restore checkpoint
        if self.saver.restore_ckpt:
            logger.info(f'Restore from {self.saver.restore_ckpt}')
            self.saver.load_checkpoint()
        else:
            logger.info('Train from scratch')

        train_loader = self.dataset['train']
        train_iterator = iter(train_loader)
        val_loader = self.dataset['val']

        # train loop
        while self.step < self.total_steps:
            try:
                data_pair = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                data_pair = next(train_iterator)

            # training
            pred, loss = self.train_step(data_pair)
            self.step = self.optimizer.iterations.numpy()

            # log the training information every n steps
            if self.step % self.log_train_info_steps == 0:
                with self.summary.as_default(step=self.step):
                    logger.info(f'Step {self.step} train loss: {loss}')
                    tf.summary.scalar('train_loss', loss)
                    tf.summary.scalar('learning_rate', self.optimizer.lr(self.optimizer.iterations))
                    tf.summary.image('pred', [pred[0] / 255.0])
                    tf.summary.image(
                        'input',
                        [data_pair[0][0, -2, ...] / 255.0, data_pair[0][0, -1, ...] / 255.0]
                    )
                    tf.summary.image('target', [data_pair[1][0, -1, ...] / 255.0])
                self.summary.flush()

            # save checkpoint every n steps
            if self.step % self.keep_ckpt_steps == 0:
                self.saver.save_checkpoint()

            # validation and log the validation results n steps
            if self.step % self.valid_steps == 0:
                for metric_name in self.metric_functions:
                    self.metric_functions[metric_name].reset()

                for data_pair in val_loader:
                    self.test_step(data_pair)
                    break

                # log the validation results
                self.log_metric('Val_')

        # save the checkpoint after finishing training
        self.saver.save_checkpoint()

    def test(self):
        """Evaluate the model."""
        self.register_evaluation()
        self.summary = tf.summary.create_file_writer(self.log_dir)

        # restore checkpoint
        logger.info(f'Restore from {self.saver.restore_ckpt}')
        self.saver.load_checkpoint()

        val_loader = self.dataset['val']

        save_path = pathlib.Path(self.log_dir) / 'output'
        save_path.mkdir(exist_ok=True)
        for i, data_pair in enumerate(val_loader):
            pred_tensors = self.test_step(data_pair)
            for j, pred_tensor in enumerate(pred_tensors):
                tf.keras.utils.save_img(
                    save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_tensor[0]
                )

        # log the evaluation results
        self.log_metric('Test_')
