# Requirements

## Python version

Python 3.9 or later. Your Python version also needs a TensorFlow release that supports it: check
[TensorFlow's installation guide](https://www.tensorflow.org/install/pip) if pip can't find
TensorFlow. The package's automated tests run on Python 3.10 to 3.13.

## Dependencies

| Package | Version |
|---|---|
| `tensorflow` | 2.16 or later |
| `numpy` | Any |
| `Pillow` | Any |

`pip install teachable-machine` installs them for you.

## TensorFlow and Keras

TensorFlow 2.16 and later come with Keras 3. The package handles the differences that stop
Teachable Machine's exported models from loading there. See
[TensorFlow Compatibility](../guides/compatibility.md).
