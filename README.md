# Teachable Machine
_By: [Meqdad Darwish](https://github.com/MeqdadDev)_


<p align="center">
<picture>
  <img alt="Teachable Machine Package Logo" src="logo.png" width="50%" height="50%" >
</picture>
</p>

[![Downloads](https://static.pepy.tech/badge/teachable-machine)](https://pepy.tech/project/teachable-machine)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PyPI](https://img.shields.io/pypi/v/teachable-machine)](https://pypi.org/project/teachable-machine/)

A Python package designed to simplify the integration of exported models from Google's [Teachable Machine](https://teachablemachine.withgoogle.com/) platform into various environments.
This tool was specifically crafted to work seamlessly with Teachable Machine, making it easier to implement and use your trained models.

Source Code is published on [GitHub](https://github.com/MeqdadDev/teachable-machine)

Read more about the project (requirements, installation, examples and more) in the [Documentation Website](https://meqdaddev.github.io/teachable-machine/) 

## Supported Classifiers

**Image Classification**: use exported keras model from Teachable Machine platform.

## Compatibility with recent TensorFlow/Keras releases

Teachable Machine's exported `.h5` models embed a legacy `DepthwiseConv2D` layer config that current Keras (Keras 3, bundled by default since TensorFlow 2.16) rejects with an error such as:

```
TypeError: Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}
```

Some exports also save the model as a `Sequential` wrapping nested `Sequential`/`Functional` submodels, a shape Keras 3's legacy H5 loader mis-rebuilds, which previously surfaced as a misleading `FileNotFoundError: Model file not found`.

Since `v1.3.0`, this package patches the model loader to handle both cases, so exported models load correctly on up-to-date TensorFlow/Keras installs, with no need to pin an old TensorFlow version. It also fixes prediction-annotation crashes on Windows / recent Pillow versions (`show_prediction_on_image`). See [issue #2](https://github.com/MeqdadDev/teachable-machine/issues/2) and the [changelog](https://meqdaddev.github.io/teachable-machine/changelog/) for background.

## Requirements

``` Python >= 3.9 ```

## How to install package

```bash
pip install teachable-machine
```

## Example

An example for teachable machine package with OpenCV:

```python
from teachable_machine import TeachableMachine
import cv2 as cv

cap = cv.VideoCapture(0)
model = TeachableMachine(model_path="keras_model.h5",
                         labels_file_path="labels.txt")

image_path = "screenshot.jpg"

while True:
    _, img = cap.read()
    cv.imwrite(image_path, img)

    result, resultImage = model.classify_and_show(image_path)

    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    print("class_confidence:", result["class_confidence"])

    print("predictions:", result["predictions"])

    cv.imshow("Video Stream", resultImage)

    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break
    
cap.release()
cv.destroyAllWindows()
```

Values of `result` are assigned based on the content of `labels.txt` file.

For more; take a look on [these examples](https://meqdaddev.github.io/teachable-machine/codeExamples/)

### Links:

- [Documentation](https://meqdaddev.github.io/teachable-machine)

- [PyPI](https://pypi.org/project/teachable-machine/)

- [Source Code](https://github.com/MeqdadDev/teachable-machine)

- [Teachable Machine Platform](https://teachablemachine.withgoogle.com/)
