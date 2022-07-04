# Real-Time Video Super-Resolution on Mobile [<img align="right" src="webpage/logo_MTK.png" width="15%">](https://www.mediatek.com/)

## Overview

[**[Challenge Website]**](https://codalab.lisn.upsaclay.fr/competitions/1756) [**[Workshop Website]**](http://ai-benchmark.com/workshops/mai/2022/)

This repository provides the implementation of the baseline model, **Mobile RRN**, for the [***Real-Time Video Super-Resolution*** Challenge](https://codalab.lisn.upsaclay.fr/competitions/1756) in [*Mobile AI (MAI) Workshop @ CVPR 2022*](http://ai-benchmark.com/workshops/mai/2022/) & [*Advances in Image Manipulation (AIM) Workshop @ ECCV 2022*](https://data.vision.ee.ethz.ch/cvl/aim22/). Mobile RRN is a recurrent network for video super-resolution to run on mobile. And it is modified from RRN with reducing channels and not using previous output information.

### Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset preparation](#dataset-preparation)
- [Training and validation](#training-and-testing)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Validation](#validation)
- [Testing](#testing)
- [Convert to tflite](#convert-to-tflite)
- [TFLite inference on mobile](#tflite-inference-on-mobile)
- [Folder structure](#folder-structure)
- [Model optimization](#model-optimization)
- [Reference](#reference)
- [License](#license)

---

### Requirements

- Python: 3.6, 3.7
- Python packages: numpy, imageio and pyyaml 
- [TensorFlow >= 2.6.0](https://www.tensorflow.org/install/) + [CUDA cuDNN](https://developer.nvidia.com/cudnn)
- GPU for training (e.g., Nvidia GeForce RTX 3090)

[[back]](#contents)

---

### Dataset preparation

- Download REDS dataset and extract it into `data` folder.

  <sub>The REDS dataset folder should contain three subfolders: `train/`, `val/` and `test/`. </sub>
  <sub>Please find the download links to above files in **MAI'22 Real-Time Video Super-Resolution Challenge** [website](https://codalab.lisn.upsaclay.fr/competitions/1756) (registration needed). </sub>

[[back]](#contents)

---

### Training and Validation

#### Configuration

Before training and testing, please make sure the fields in `config.yml` is properly set.

```yaml
log_dir: snapshot -> The directory which records logs and checkpoints. 

dataset:
    dataloader_settings: -> The setting of different splits dataloader.
        train:
            batch_size: 4
            drop_remainder: True
            shuffle: True
            num_parallel_calls: 6
        val:
            batch_size: 1
    data_dir: data/ -> The directory of REDS dataset.
    degradation: sharp_bicubic -> The degradation of images.
    train_frame_num: 10 -> The number of image frame(s) for per training step.
    test_frame_num: 100 -> The number of image frame(s) for per testing step.
    crop_size: 64 -> The height and width of cropped patch.

model:
    path: model/mobile_rrn.py -> The path of model file.
    name: MobileRRN -> The name of model class.

learner:
    general:
        total_steps: 1500000 -> The number of training steps.
        log_train_info_steps: 100 -> The frequency of logging training info.
        keep_ckpt_steps: 10000 -> The frequency of saving checkpoint.
        valid_steps: 100000 -> The frequency of validation.

    optimizer: -> Define the module name and setting of optimizer
        name: Adam
        beta_1: 0.9
        beta_2: 0.999

    lr_scheduler: -> Define the module name and setting of learning rate scheduler
        name: ExponentialDecay
        initial_learning_rate: 0.0001
        decay_steps: 1000000
        decay_rate: 0.1
        staircase: True

    saver:
        restore_ckpt: null -> The path to checkpoint where would be restored from.
```

#### Training

To train the model, use the following command:

```bash
python run.py --process train --config_path config.yml
```

The main arguments are as follows:

>```process``` : &nbsp; Process type should be train or test.<br/>
>```config_path``` : &nbsp; Path of yml config file of the application.<br/>

After training, the checkpoints will be produced in `log_dir`.

#### Validation

To valid the model, use the following command:

```bash
python run.py --process test --config_path config.yml
```

After testing, the output images will be produced in `log_dir/output`.

[[back]](#contents)

---

### Testing

To generate testing outputs, use the following command:

```bash
python generate_output.py --model_path model/mobile_rrn.py --model_name MobileRRN --ckpt_path snapshot/ckpt-* --data_dir REDS/test/test_sharp_bicubic/X4/ --output_dir results
```

The main arguments are as follows:

>```model_path``` : &nbsp; Path of model file.<br/>
>```model_name``` : &nbsp; Name of model class.<br/>
>```ckpt_path``` : &nbsp; Path of checkpoint.<br/>
>```data_dir``` : &nbsp; Directory of testing frames in REDS dataset.<br/>
>```output_dir``` : &nbsp; Directory for saving output images.<br/>

[[back]](#contents)

---

### Convert to tflite

To convert the keras model to tflite, use the following command:

```bash
python convert.py --model_path model/mobile_rrn.py --model_name MobileRRN --input_shapes 1,320,180,6:1,320,180,16 --ckpt_path snapshot/mobile_rrn_16/ckpt-* --output_tflite model.tflite
```

The main arguments are as follows:

>```model_path``` : &nbsp; Path of model file.<br/>
>```model_name``` : &nbsp; Name of model class.<br/>
>```input_shape``` : &nbsp; Series of the input shapes split by \`:\`.<br/>
>```ckpt_path``` : &nbsp; Path of checkpoint.<br/>
>```output_tflite``` : &nbsp; Path of output tflite.<br/>

[[back]](#contents)

---

### TFLite inference on Mobile

We provide two ways to evaluate the mobile performance of your TFLite models:

- [AI benchmark](https://ai-benchmark.com/): An app allowing you to load your model and run it locally on your own Android devices with various acceleration options (e.g. CPU, GPU, APU, etc.).
- [TFLite Neuron Delegate](https://github.com/MediaTek-NeuroPilot/tflite-neuron-delegate): You can build MediaTek's neuron delegate runner by yourself.

[[back]](#contents)

---

### Folder structure

```text
│
├── data/ -> The directory places the REDS dataset
├── dataset/
|   ├── dataset_builder.py -> Builds the dataset loader.
|   ├── reds.py -> Define the class of REDS dataset.
|   └── transform.py -> Define the transform functions for augmentation.
├── learner/
|   ├── learner.py -> Define the learner for training and testing.
|   ├── metric.py -> Implement the metric functions.
|   └── saver.py -> Define the saver to save and load checkpoints.
├── model/
|   └── mobile_rrn.py -> Define Mobile RRN architecture.
├── snapshot/ -> The directory which records logs and checkpoints. 
├── util/
|   ├── common_util.py -> Define the utility functions for general purpose.
|   ├── constant_util.py -> Global constant definitions.
|   ├── logger.py -> Define logging utilities.
|   └── plugin.py -> Define plugin utilities.
├── config.yml -> Configuration yaml file.
├── convert.py -> Convert keras model to tflite.
└── run.py -> Generic main function for VSR experiments.
```

[[back]](#contents)

---

### Model Optimization

To make your model run faster on device, please fulfill the preference of network operations as much as possible to leverage the great power of AI accelerator.
You may also find some optimization hint from our paper: [Deploying Image Deblurring across Mobile Devices: A Perspective of Quality and Latency](https://arxiv.org/abs/2004.12599)

[[back]](#contents)

---

### Reference

Revisiting Temporal Modeling for Video Super-resolution (RRN) [[Github]](https://github.com/junpan19/RRN) [[Paper]](https://arxiv.org/abs/2008.05765)

[[back]](#contents)

---

### License

Mediatek License: [Mediatek Apache License 2.0](LICENSE)

[[back]](#contents)
