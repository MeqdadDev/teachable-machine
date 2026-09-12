# Welcome to Teachable Machine Package
_By: [Meqdad Darwish](https://github.com/MeqdadDev)_

<p align="center">
<picture>
  <img alt="Teachable Machine Package Logo" src="logo.png" width="80%" height="80%" >
</picture>
</p>

[![Downloads](https://static.pepy.tech/badge/teachable-machine)](https://pepy.tech/project/teachable-machine)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PyPI](https://img.shields.io/pypi/v/teachable-machine)](https://pypi.org/project/teachable-machine/)

## Description
A Python package designed to simplify the integration of exported models from Google's [Teachable Machine](https://teachablemachine.withgoogle.com/) platform into various environments.
This tool was specifically crafted to work seamlessly with Teachable Machine, making it easier to implement and use your trained models.

Source Code is published on [GitHub](https://github.com/MeqdadDev/teachable-machine)

## Table Of Contents

1. [How-To Guide](how-to-guide.md)
2. [Requirements](requirements.md)
3. [Code Examples](codeExamples.md)
4. [Explanation](explanation.md)
5. [Changelog](changelog.md)
6. [Contributing](contribution.md)
7. [Releasing](releasing.md)

## Supported Classifiers

**Image Classification**

## Compatibility with recent TensorFlow/Keras releases

Teachable Machine's exported `.h5` models embed a legacy `DepthwiseConv2D` layer config that current Keras (Keras 3, bundled by default since TensorFlow 2.16) rejects with an error such as:

```
TypeError: Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}
```

Some exports also save the model as a `Sequential` wrapping nested `Sequential`/`Functional` submodels, a shape Keras 3's legacy H5 loader mis-rebuilds, which previously surfaced as a misleading `FileNotFoundError: Model file not found`.

Since `v1.3.1`, this package patches the model loader to handle both cases, so exported models load correctly on up-to-date TensorFlow/Keras installs, with no need to pin an old TensorFlow version. It also fixes prediction-annotation crashes on Windows / recent Pillow versions (`show_prediction_on_image`). See [issue #2](https://github.com/MeqdadDev/teachable-machine/issues/2) and the [changelog](changelog.md) for background.

## Links

- [PyPI](https://pypi.org/project/teachable-machine/)

- [Source Code](https://github.com/MeqdadDev/teachable-machine)

- [Teachable Machine Platform](https://teachablemachine.withgoogle.com/)
