"""Define Mobile RRN architecture.

Mobile RRN is a lite version of Revisiting Temporal Modeling (RRN) which is a recurrent network for
video super-resolution to run on mobile.

Each Mobile RRN cell firstly concatenate input sequence LR frames and hidden state.
Then, forwarding it through several residual blocks to output prediction and update hidden state.

Reference paper https://arxiv.org/abs/2008.05765
Reference github https://github.com/junpan19/RRN/
"""

import tensorflow as tf


class MobileRRN(tf.keras.Model):
    """Implement Mobile RRN architecture.

    Attributes:
        scale: An `int` indicates the upsampling rate.
        base_channels: An `int` represents the number of base channels.
    """

    def __init__(self,):
        """Initialize `RRN`."""
        super().__init__()
        in_channels = 3
        out_channels = 3
        block_num = 5  # the number of residual block in RNN cell

        self.base_channels = 16
        self.scale = 4

        # first conv
        self.conv_first = tf.keras.layers.Conv2D(
            self.base_channels, kernel_size=3, strides=1, padding='SAME', activation='relu'
        )
        self.recon_trunk = make_layer(ResidualBlock, block_num, base_channels=self.base_channels)

        self.conv_last = tf.keras.layers.Conv2D(
            self.scale * self.scale * out_channels, kernel_size=3, strides=1, padding='SAME'
        )
        self.conv_hidden = tf.keras.layers.Conv2D(
            self.base_channels, kernel_size=3, strides=1, padding='SAME', activation='relu'
        )

    def call(self, inputs, training=False):
        """Forward the given input.

        Args:
            inputs: An input `Tensor` and an `Tensor` represents the hidden state.
            training: A `bool` indicates whether the current process is training or testing.

        Returns:
            An output `Tensor`.
        """
        x, hidden = inputs
        x1 = x[:, :, :, :3]
        x2 = x[:, :, :, 3:]
        _, h, w, _ = x1.shape.as_list()

        x = tf.concat((x1, x2, hidden), axis=-1)
        out = self.conv_first(x)
        out = self.recon_trunk(out)
        hidden = self.conv_hidden(out)
        out = self.conv_last(out)

        out = tf.nn.depth_to_space(out, self.scale)
        bilinear = tf.image.resize(x2, size=(h * self.scale, w * self.scale))
        out = out + bilinear

        if not training:
            out = tf.clip_by_value(out, 0, 255)

        return out, hidden


class ResidualBlock(tf.keras.Model):
    """Residual block."""

    def __init__(self, base_channels):
        """Initialize `ResidualBlock`.

        Args:
            base_channels: An `int` represents the number of base channels.
        """
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            base_channels, kernel_size=3, strides=1, padding='SAME', activation='relu'
        )
        self.conv2 = tf.keras.layers.Conv2D(base_channels, kernel_size=3, strides=1, padding='SAME')

    def call(self, x):
        """Forward the given input.

        Args:
            x: An input `Tensor`.

        Returns:
            An output `Tensor`.
        """
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return identity + out


def make_layer(basic_block, block_num, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block: A `nn.module` represents the basic block.
        block_num: An `int` represents the number of blocks.

    Returns:
        An `nn.Sequential` stacked blocks.
    """
    model = tf.keras.Sequential()
    for _ in range(block_num):
        model.add(basic_block(**kwarg))
    return model
