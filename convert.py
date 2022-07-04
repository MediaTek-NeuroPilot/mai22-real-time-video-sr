"""Convert keras model to tflite."""

import argparse

import tensorflow as tf

from util import plugin


def _parse_argument():
    """Return arguments for conversion."""
    parser = argparse.ArgumentParser(description='Conversion.')
    parser.add_argument('--model_path', help='Path of model file.', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model class.', type=str, required=True)
    parser.add_argument(
        '--input_shapes', help='Series of the input shapes split by `:`.', required=True
    )
    parser.add_argument('--ckpt_path', help='Path of checkpoint.', type=str, required=True)
    parser.add_argument('--output_tflite', help='Path of output tflite.', type=str, required=True)

    args = parser.parse_args()

    return args


def main(args):
    """Run main function for converting keras model to tflite.

    Args:
        args: A `dict` contain augments.
    """
    # prepare model
    model_builder = plugin.plugin_from_file(args.model_path, args.model_name, tf.keras.Model)
    model = model_builder()

    # load checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.ckpt_path).expect_partial()

    # build model with fake input data
    input_tensors = []
    for input_shape in args.input_shapes.split(':'):
        input_shape = list(map(int, input_shape.split(',')))
        input_tensors.append(tf.random.normal(input_shape))
    model(input_tensors)

    # convert the keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # save the tflite
    with open(args.output_tflite, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    arguments = _parse_argument()
    main(arguments)