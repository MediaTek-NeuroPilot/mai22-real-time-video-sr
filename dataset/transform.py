"""Define transform methods.

classes:
    NormalizePair: Normalize the given data pair with mean and standard deviation.
    RandomHorizontalFlipPair: Horizontally flip the given data pair randomly.
    RandomVerticalFlipPair: Vertically flip the given data pair randomly.
    RandomRot90Pair: Rotate 90 degrees on the given data pair randomly.
    RandomCropSRPair: Crop a patch from given SR data pair randomly.
"""
import collections

import tensorflow as tf

# namedtuple for convenience.
TFDataPair = collections.namedtuple('TFDataPair', ['image', 'label'])


class NormalizePair():
    """Normalize the given data pair with mean and standard deviation.

    Attributes:
        input_mean, input_std, target_mean, target_std:
            A `float`, `list`, `tuple`, `1D-array` represents mean or standard deviation for
                input or target.
            If it's not a single number, it should match the number of channel.
            For instance, for the RGB input_img, the value could be a single number, or 3 numbers.
    """

    def __init__(self, image_mean=0., image_std=1., label_mean=0., label_std=1.):
        """Initialize attributes.

        Args:
            input_mean: Please refer to Attributes. Defaults to 0.0.
            input_std: Please refer to Attributes. Defaults to 1.0.
            target_mean: Please refer to Attributes. Defaults to 0.0.
            target_std: Please refer to Attributes. Defaults to 1.0.
        """
        self.image_mean = image_mean
        self.image_std = image_std
        self.label_mean = label_mean
        self.label_std = label_std

    def __call__(self, data_pair):
        """Use `tf.Tensor` basic operators to run normalization on `tf.Tensor`."""
        image = (tf.cast(data_pair.image, tf.float32) - self.image_mean) / self.image_std
        label = (tf.cast(data_pair.label, tf.float32) - self.label_mean) / self.label_std
        return TFDataPair(image, label)


class RandomHorizontalFlipPair():
    """Horizontally flip the given data pair randomly.

    Should apply flipping on both image and label simultaneously.
    """

    def __call__(self, data_pair):
        """Use `tf.image.flip_left_right` to run hflip on `tf.Tensor`."""

        def _flip_left_right():
            """Perform hflip."""
            image = tf.image.flip_left_right(data_pair.image)
            label = tf.image.flip_left_right(data_pair.label)
            return TFDataPair(image, label)

        return tf.compat.v1.cond(tf.random.uniform([]) > 0.5, _flip_left_right, lambda: data_pair)


class RandomVerticalFlipPair():
    """Vertically flip the given data pair randomly.

    Should apply flipping on both image and label simultaneously.
    """

    def __call__(self, data_pair):
        """Use `tf.image.flip_up_down` to run vflip on `tf.Tensor`."""

        def _flip_up_down():
            """Perform vflip."""
            image = tf.image.flip_up_down(data_pair.image)
            label = tf.image.flip_up_down(data_pair.label)
            return TFDataPair(image, label)

        return tf.compat.v1.cond(tf.random.uniform([]) > 0.5, _flip_up_down, lambda: data_pair)


class RandomRot90Pair():
    """Rotate 90 degrees on the given data pair randomly.

    Could rotate one time or many times.

    Attributes:
        times: An integer, the times to rotate 90 degrees.
    """

    def __init__(self, times):
        """Initialize with rotation arguments.

        Args:
            times: See `Attributes`.
        """
        self.times = times

    def __call__(self, data_pair):
        """Use `tf.image.rot90` to run rotation on `tf.Tensor`."""

        def _rotate():
            """Perform rotation."""
            image = tf.image.rot90(data_pair.image, k=self.times)
            label = tf.image.rot90(data_pair.label, k=self.times)
            return TFDataPair(image, label)

        return tf.compat.v1.cond(tf.random.uniform([]) > 0.5, _rotate, lambda: data_pair)


class RandomCropSRPair():
    """For super resolution data, crop a patch from given data pair randomly.

    Since the output of super resolution is larger than input. Will use `scale` to adjust the crop
    size of input-image and output-ground-truth.

    Attributes:
        lr_crop_size: A `tuple` represents desired output size (height, width) of
            low-resolution input.
            The crop size of high-resolution target is `lr_crop_size` * `scale`.
        scale: An `int` represents the scale ratio between the high-resolution target to
            low-resolution input.
    """

    def __init__(self, lr_crop_size, scale):
        """Initialize the arguments.

        Args:
            lr_crop_size: Please refer to Attributes.
            scale: Please refer to Attributes.
        """
        self.lr_crop_size = lr_crop_size
        self.scale = scale

    def __call__(self, data_pair):
        """Use `tf.image.random_crop` to crop on `tf.Tensor`."""
        # get parameters for the random crop
        height, width = tf.shape(data_pair.image)[1], tf.shape(data_pair.image)[2]

        h_crop, w_crop = self.lr_crop_size
        h_start = tf.random.uniform([], maxval=height - h_crop + 1, dtype=tf.int32)
        w_start = tf.random.uniform([], maxval=width - w_crop + 1, dtype=tf.int32)

        # crop a patch as LR input, and then crop a larger patch (x scale) as HR output
        image = tf.image.crop_to_bounding_box(data_pair.image, h_start, w_start, h_crop, w_crop)
        label = tf.image.crop_to_bounding_box(
            data_pair.label,
            h_start * self.scale,
            w_start * self.scale,
            h_crop * self.scale,
            w_crop * self.scale
        )
        return TFDataPair(image, label)


class RandomCropPair():
    """Crop a patch from given data pair randomly.

    Attributes:
        lr_crop_size: A `tuple` represents desired output size (height, width).
    """

    def __init__(self, crop_size):
        """Initialize the arguments.

        Args:
            crop_size: Please refer to Attributes.
        """
        self.crop_size = crop_size

    def __call__(self, data_pair):
        """Use `tf.image.random_crop` to crop on `tf.Tensor`."""
        # get parameters for the random crop
        height, width = tf.shape(data_pair.image)[0], tf.shape(data_pair.image)[1]
        h_crop, w_crop = self.crop_size
        h_start = tf.random.uniform([], maxval=height - h_crop + 1, dtype=tf.int32)
        w_start = tf.random.uniform([], maxval=width - w_crop + 1, dtype=tf.int32)

        # crop a patch as LR input, and then crop a larger patch (x scale) as HR output
        image = tf.image.crop_to_bounding_box(data_pair.image, h_start, w_start, h_crop, w_crop)
        label = tf.image.crop_to_bounding_box(data_pair.label, h_start, w_start, h_crop, w_crop)
        return TFDataPair(image, label)
