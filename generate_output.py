"""Generate testing output."""

import argparse
import pathlib

import imageio
import numpy as np
import tensorflow as tf

from util import plugin


def _parse_argument():
    """Return arguments for conversion."""
    parser = argparse.ArgumentParser(description='Testing.')
    parser.add_argument('--model_path', help='Path of model file.', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model class.', type=str, required=True)
    parser.add_argument('--ckpt_path', help='Path of checkpoint.', type=str, required=True)
    parser.add_argument(
        '--data_dir', help='Directory of testing frames in REDS dataset.', type=str, required=True
    )
    parser.add_argument(
        '--output_dir', help='Directory for saving output images.', type=str, required=True
    )

    args = parser.parse_args()

    return args


def main(args):
    """Run main function for converting keras model to tflite.

    Args:
        args: A `dict` contain augments.
    """
    # prepare dataset
    data_dir = pathlib.Path(args.data_dir)

    # prepare model
    model_builder = plugin.plugin_from_file(args.model_path, args.model_name, tf.keras.Model)
    model = model_builder()

    # load checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.ckpt_path).expect_partial()

    save_path = pathlib.Path(args.output_dir)
    save_path.mkdir(exist_ok=True)
    # testing
    for i in range(30):
        for j in range(100):
            if j == 0:
                input_image = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                b, h, w, _ = input_image.shape
                input_tensor = tf.concat([input_image, input_image], axis=-1)
                hidden_state = tf.zeros([b, h, w, model.base_channels])
                pred_tensor, hidden_state = model([input_tensor, hidden_state], training=False)
            else:
                input_image_1 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j-1).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                input_image_2 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                input_tensor = tf.concat([input_image_1, input_image_2], axis=-1)
                pred_tensor, hidden_state = model([input_tensor, hidden_state], training=False)

            imageio.imwrite(save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_tensor[0])


if __name__ == '__main__':
    arguments = _parse_argument()
    main(arguments)