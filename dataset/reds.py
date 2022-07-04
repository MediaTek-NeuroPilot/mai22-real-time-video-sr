"""Define the class of REDS dataset.

The Realistic and Dynamic Scenes dataset (REDS) contains 300 snippet.
240 snippets for training, 30 snippets for validation and 30 snippets for testing.
There are 100 frames with 1280x720 resolution in each snippet.
It was used in the NTIRE 2019 and NTIRE 2020 Challenges (video deblurring and super-resolution).

REDS official website: https://seungjunnah.github.io/Datasets/reds.html
"""
import copy
import pathlib

import imageio
import numpy as np
import tensorflow as tf

from dataset import transform
from util import logger
from util.common_util import check_config


class REDS():
    """Implement the class of REDS dataset for deblur and sr application.

    Here's the expected structure.

        /path/to/REDS/
        │
        ├── train/
        │   ├── train_sharp/
        │   │   ├── 000/
        │   │   ├── ...
        │   │   └── 239/
        │   │        ├── 00000000.png
        │   │        ├── ...
        │   │        └── 00000099.png
        |   ├── train_blur/
        │   ├── train_sharp_bicubic/
        │   │   └── X4/
        │   │       ├── 000/
        |   │       ├── ...
        │   │       └── 239/
        │   │           ├── 00000000.png
        │   │           ├── ...
        │   │           └── 00000099.png
        │   └── train_blur_bicubic/
        └── val/
            ├── val_sharp/
            │   ├── 000/
            │   ├── ...
            │   └── 029/
            │        ├── 00000000.png
            │        ├── ...
            │        └── 00000099.png
            ├── val_blur/
            ├── val_sharp_bicubic/
            │   └── X4/
            │       ├── 000/
            │       ├── ...
            │       └── 029/
            │           ├── 00000000.png
            │           ├── ...
            │           └── 00000099.png
            └── val_blur_bicubic/
        (all sub-dir structure is similar)

    Attributes:
        SPLITS: A `list` indicates all kinds of splits ('train', 'val') in the dataset.
        DEGRADATIONS: A `list` indicates all kinds of degradations ('blur', 'blur_bicubic',
            'sharp_bicubic') in the dataset.

        image_list: A `list` contains filenames of image sequence of frames.
        label_list: A `list` contains filenames of label sequence of frames.
        transform: A `list` of transforms for augmentation and preprocessing.
    """

    SPLITS = ['train', 'val']
    DEGRADATIONS = ['blur', 'blur_bicubic', 'sharp_bicubic']

    def __init__(self, dataset_config, split):
        """Initialize `REDS` and prepare image list and label list.

        A sample `dataset_config` may look like the following
            dataset_config = {
                'data_dir': '/path/to/reds/',
                'degradation': 'sharp_bicubic',
                'train_frame_num': 10,
                'test_frame_num': 100,
                'crop_size': 224
            }
            data_dir: A `str` represents the directory of REDS dataset.
            degradation: A `str` indicates the chosen degradation of images.
            train_frame_num: An `int` represents the number of image frame(s) for training.
                It should be 1 for simple training, and > 1 for recurrent sequence-to-sequence
                training. Defaults to 1.
            test_frame_num: An `int` represents the number of image frame(s) for testing.
                Defaults to 1.
            crop_size: An `int` represents the height and width of cropped patch. Defaults to 224.

        Args:
            dataset_config: A `dict` contains the configuration of the dataset.
            split: A `str` represents the subset of dataset (train/val).

        Raises:
            KeyError: If any required configuration is not provided.
            ValueError:
                1. If the degradation or the split is unexpected.
                2. If some keys in the configuration are not supported.
                3. If `data_dir` is not a directory.
                4. If the structure of the dataset is wrong.
        """
        logger.check(split in self.SPLITS, f'Unexpected dataset split: `{split}`. ' + \
                'Supported splits are: `train` and `val`')

        self.split = split
        with check_config(copy.deepcopy(dataset_config)) as config:
            data_dir = pathlib.Path(config.pop('data_dir'))
            degradation = config.pop('degradation')
            self.degradation = degradation
            train_frame_num = config.pop('train_frame_num', 1)
            test_frame_num = config.pop('test_frame_num', 1)
            self.crop_size = config.pop('crop_size', 224)

        logger.check(degradation in self.DEGRADATIONS,
                f'Unexpected degradation: `{degradation}`. ' + \
                'Supported degradations are: `blur`, `blur_bicubic` and `sharp_bicubic`')
        logger.check(data_dir.is_dir(), f'`{data_dir}` is not a directory.')

        # parse all data files
        image_dir = data_dir / split / f'{split}_{degradation}'
        if 'bicubic' in degradation:
            image_dir = image_dir / 'X4'
        label_dir = data_dir / split / f'{split}_sharp'
        filenames = sorted(image_dir.glob('**/*.png'))

        self.image_list, self.label_list = [], []
        frame_num = train_frame_num if split == 'train' else test_frame_num
        # add eligible sequence of frames in to list
        for filename in filenames:
            frame_idx = int(filename.stem)
            video_idx = filename.parts[-2]
            image_sequence = _get_sequence(
                image_dir / video_idx, filename.suffix, frame_idx, frame_num
            )
            label_sequence = _get_sequence(
                label_dir / video_idx, filename.suffix, frame_idx, frame_num
            )
            if image_sequence and label_sequence:
                self.image_list.append(image_sequence)
                self.label_list.append(label_sequence)
        logger.check(
            len(self.image_list) and len(self.label_list),
            'The structure of the dataset is wrong, please refer to the docstring of REDS.'
        )

    def build(
        self,
        batch_size,
        num_parallel_calls=None,
        shuffle=False,
        buffer_size=None,
        drop_remainder=False,
    ):
        """Build Tensorflow dataset.

        Args:
            batch_size: An integer specifying the batch size.
            num_parallel_calls: An integer specifying the number of elements to process
                in parallel. Defaults to None.
            shuffle: A boolean indicating whether to shuffle the dataset. Defaults to False.
            buffer_size: An integer specifying the buffer size for shuffle.
                Defaults to batch size multiplied by 20.
            drop_remainder: A boolean indicating whether to drop remained elements.
                Defaults to False.

        Returns:
            A `tf.data.Iterator` instance.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            ([paths for paths in self.image_list], [paths for paths in self.label_list])
        )

        dataset = dataset.map(self._preprocess, num_parallel_calls=num_parallel_calls)

        if shuffle:
            buffer_size = buffer_size if buffer_size is not None \
                else batch_size * 20
            dataset = dataset.shuffle(buffer_size)
        if self.split == 'train':
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def _preprocess(self, image_paths, label_paths):
        """Preprocess dataset.

        Args:
            image_paths: A `list` of `pathlib.Path` represents the filepath of images.
            label_paths: A `list` of `pathlib.Path` represents the filepath of ground truth images.

        Return:
            A image and label pair where label is the ground-truth of the image.
        """
        image_tensor, label_tensor = tf.compat.v1.py_func(
            _loader, (image_paths, label_paths), (tf.float32, tf.float32))
        image_tensor.set_shape([None, None, None, 3])
        label_tensor.set_shape([None, None, None, 3])

        data_pair = transform.TFDataPair(image_tensor, label_tensor)
        # augmentation
        if self.split == 'train':
            if 'bicubic' in self.degradation:
                data_pair = transform.RandomCropSRPair((self.crop_size, self.crop_size),
                                                       scale=4)(data_pair)
            else:
                data_pair = transform.RandomCropPair((self.crop_size, self.crop_size))(data_pair)
            data_pair = transform.RandomHorizontalFlipPair()(data_pair)
            data_pair = transform.RandomVerticalFlipPair()(data_pair)
            data_pair = transform.RandomRot90Pair(times=1)(data_pair)

        return data_pair


def _loader(image_paths, label_paths):
    """Load an image and label pair with given paths.

    Args:
        image_paths: A `list` of `pathlib.Path` represents the filepath of images.
        label_paths: A `list` of `pathlib.Path` represents the filepath of ground truth images.

    Return:
        A image and label pair where label is the ground-truth of the image.
    """
    # read images
    image_imgs = [imageio.imread(f.decode('utf-8')) for f in image_paths]
    label_imgs = [imageio.imread(f.decode('utf-8')) for f in label_paths]

    image = np.stack(image_imgs,
                     axis=0) if len(image_imgs) > 1 else np.expand_dims(image_imgs[0], axis=0)
    label = np.stack(label_imgs,
                     axis=0) if len(label_imgs) > 1 else np.expand_dims(label_imgs[0], axis=0)

    return transform.TFDataPair(image.astype(np.float32), label.astype(np.float32))


def _get_sequence(data_dir, extension, frame_idx, frame_num):
    """Get filenames of sequence of frames.

    Args:
        data_dir: A `PosixPath` represents the data directory.
        extension: A `str` represents the file extension.
        frame_idx: An `int` represents the index of the key frame.
        frame_num: An `int` represents the number of label frame(s).

    Returns:
        sequence: A `list` contains filenames of sequence of frames.
            Return `None` if no frame exists.
    """
    sequence = []
    for idx in range(frame_idx, frame_idx + frame_num):
        filename = data_dir / (str(idx).zfill(8) + extension)
        if filename.exists():
            sequence.append(str(filename))
        else:
            return None
    return sequence
