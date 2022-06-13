"""Implement metric functions."""

import abc

import tensorflow as tf


class MetricBase(metaclass=abc.ABCMeta):
    """Implement the abstract class of metric function."""

    @abc.abstractmethod
    def reset(self):
        """Reset all of the metric state variables."""

    @abc.abstractmethod
    def update(self, pred, target):
        """Compute and accumulate metric statistics."""

    @abc.abstractmethod
    def get_result(self):
        """Return the metric value tensor."""

    @abc.abstractmethod
    def get_count(self):
        """Return the the number of calculations."""


class PSNR(MetricBase):
    """Implement peak signal-to-noise ratio (psnr) metric.

    Attributes:
        data_range: An `int` represents the dynamic range of the images, which is the
            difference between the maximum and the minimum allowed values of the image-type.
        count: An `int` represents the number of images been calculated, used for averaging psnr.
        value: A `float` represents the accumulated psnr value.
    """

    def __init__(self, data_range):
        """Initialize attributes.

        Args:
            data_range: Please refer to Attributes.
        """
        self.data_range = data_range
        self.reset()

    def reset(self):
        """Reset count and value to 0."""
        self.count = 0
        self.value = 0.0

    def update(self, pred, target):
        """Compute the psnr value and accumulate it.

        Args:
            pred: A `tf.tensor` represents the predicted image with shape (N, H, W, C).
            target: A `tf.tensor` represents the target image with shape (N, H, W, C).
        """
        psnr = tf.image.psnr(pred, target, max_val=self.data_range)
        self.value += psnr[0]
        self.count += 1

    def get_result(self):
        """Return the mean psnr value."""
        return self.value / self.count

    def get_count(self):
        """Return the number of images been calculated."""
        return self.count


class SSIM(MetricBase):
    """Implement structural similarity (ssim) metric.

    Attributes:
        data_range: An `int` represents the dynamic range of the target image, which is the
            difference between the maximum and the minimum allowed values of the image-type.
        count: An `int` represents the number of images been calculated, used for averaging ssim.
        value: A `float` represents the accumulated ssim value.
    """

    def __init__(self, data_range):
        """Initialize attributes.

        Args:
            data_range: Please refer to Attributes.
        """
        self.data_range = data_range
        self.reset()

    def reset(self):
        """Reset count and value to 0."""
        self.count = 0
        self.value = 0.0

    def update(self, pred, target):
        """Compute the ssim value and accumulate it.

        Args:
            pred: A `tf.tensor` represents the predicted image with shape (N, H, W, C).
            target: A `tf.tensor` represents the target image with shape (N, H, W, C).
        """
        ssim = tf.image.ssim(pred, target, max_val=self.data_range)
        self.value += ssim[0]
        self.count += 1

    def get_result(self):
        """Return the mean ssim value."""
        return self.value / self.count

    def get_count(self):
        """Return the number of images been calculated."""
        return self.count
